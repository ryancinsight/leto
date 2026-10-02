"""Windows Job Object ownership for bounded subprocess trees.

Job-object completion messages are best-effort notifications, so cleanup waits
on the I/O completion port and verifies ActiveProcesses with
QueryInformationJobObject. See Microsoft's
job object <https://learn.microsoft.com/en-us/windows/win32/procthread/job-objects>,
completion-port association
<https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-jobobject_associate_completion_port>,
and accounting structure
<https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-jobobject_basic_accounting_information>
contracts.
"""

from __future__ import annotations

import ctypes
import math
import time
from contextlib import contextmanager
from ctypes import wintypes
from typing import Iterator

CREATE_SUSPENDED = 0x00000004
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
# Win32 JobObjectAssociateCompletionPortInformation and
# JobObjectBasicAccountingInformation class values.
JOB_OBJECT_ASSOCIATE_COMPLETION_PORT_INFORMATION = 7
JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION = 1
# JOB_OBJECT_MSG_ACTIVE_PROCESS_ZERO from the completion-port contract.
JOB_OBJECT_MSG_ACTIVE_PROCESS_ZERO = 4
PROCESS_ACCESS = 0x00100101
THREAD_SUSPEND_RESUME = 0x0002
TH32CS_SNAPTHREAD = 0x00000004
WAIT_TIMEOUT = 258
INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value


class WindowsJobError(RuntimeError):
    """A Windows process tree could not be owned or terminated."""


class _BasicLimitInformation(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class _IoCounters(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_uint64),
        ("WriteOperationCount", ctypes.c_uint64),
        ("OtherOperationCount", ctypes.c_uint64),
        ("ReadTransferCount", ctypes.c_uint64),
        ("WriteTransferCount", ctypes.c_uint64),
        ("OtherTransferCount", ctypes.c_uint64),
    ]


class _ExtendedLimitInformation(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _BasicLimitInformation),
        ("IoInfo", _IoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class _BasicAccountingInformation(ctypes.Structure):
    _fields_ = [
        ("TotalUserTime", ctypes.c_int64),
        ("TotalKernelTime", ctypes.c_int64),
        ("ThisPeriodTotalUserTime", ctypes.c_int64),
        ("ThisPeriodTotalKernelTime", ctypes.c_int64),
        ("TotalPageFaultCount", wintypes.DWORD),
        ("TotalProcesses", wintypes.DWORD),
        ("ActiveProcesses", wintypes.DWORD),
        ("TotalTerminatedProcesses", wintypes.DWORD),
    ]


class _AssociateCompletionPort(ctypes.Structure):
    _fields_ = [
        ("CompletionKey", ctypes.c_void_p),
        ("CompletionPort", wintypes.HANDLE),
    ]


class _ThreadEntry(ctypes.Structure):
    _fields_ = [
        ("dwSize", wintypes.DWORD),
        ("cntUsage", wintypes.DWORD),
        ("th32ThreadID", wintypes.DWORD),
        ("th32OwnerProcessID", wintypes.DWORD),
        ("tpBasePri", wintypes.LONG),
        ("tpDeltaPri", wintypes.LONG),
        ("dwFlags", wintypes.DWORD),
    ]


_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
_kernel32.CreateJobObjectW.restype = wintypes.HANDLE
_kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
_kernel32.SetInformationJobObject.restype = wintypes.BOOL
_kernel32.SetInformationJobObject.argtypes = [
    wintypes.HANDLE,
    ctypes.c_int,
    ctypes.c_void_p,
    wintypes.DWORD,
]
_kernel32.CreateIoCompletionPort.restype = wintypes.HANDLE
_kernel32.CreateIoCompletionPort.argtypes = [
    wintypes.HANDLE,
    wintypes.HANDLE,
    ctypes.c_size_t,
    wintypes.DWORD,
]
_kernel32.GetQueuedCompletionStatus.restype = wintypes.BOOL
_kernel32.GetQueuedCompletionStatus.argtypes = [
    wintypes.HANDLE,
    ctypes.POINTER(wintypes.DWORD),
    ctypes.POINTER(ctypes.c_size_t),
    ctypes.POINTER(ctypes.c_void_p),
    wintypes.DWORD,
]
_kernel32.QueryInformationJobObject.restype = wintypes.BOOL
_kernel32.QueryInformationJobObject.argtypes = [
    wintypes.HANDLE,
    ctypes.c_int,
    ctypes.c_void_p,
    wintypes.DWORD,
    ctypes.POINTER(wintypes.DWORD),
]
_kernel32.OpenProcess.restype = wintypes.HANDLE
_kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
_kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
_kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
_kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
_kernel32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
_kernel32.Thread32First.restype = wintypes.BOOL
_kernel32.Thread32First.argtypes = [wintypes.HANDLE, ctypes.POINTER(_ThreadEntry)]
_kernel32.Thread32Next.restype = wintypes.BOOL
_kernel32.Thread32Next.argtypes = [wintypes.HANDLE, ctypes.POINTER(_ThreadEntry)]
_kernel32.OpenThread.restype = wintypes.HANDLE
_kernel32.OpenThread.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
_kernel32.ResumeThread.restype = wintypes.DWORD
_kernel32.ResumeThread.argtypes = [wintypes.HANDLE]
_kernel32.TerminateJobObject.restype = wintypes.BOOL
_kernel32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
_kernel32.CloseHandle.restype = wintypes.BOOL
_kernel32.CloseHandle.argtypes = [wintypes.HANDLE]


def _error(action: str) -> WindowsJobError:
    return WindowsJobError(f"{action}: Windows error {ctypes.get_last_error()}")


def _error_code(action: str, code: int) -> WindowsJobError:
    return WindowsJobError(f"{action}: Windows error {code}")


def _close_handle(handle: object, action: str) -> None:
    if handle and not _kernel32.CloseHandle(handle):
        raise _error(action)


@contextmanager
def _owned_handle(handle: object, action: str) -> Iterator[object]:
    try:
        yield handle
    except BaseException as operation_error:
        try:
            _close_handle(handle, action)
        except WindowsJobError as close_error:
            raise WindowsJobError(f"{operation_error}; {close_error}") from operation_error
        raise
    else:
        _close_handle(handle, action)


class WindowsJob:
    """Own one suspended root process and all descendants in one OS job."""

    def __init__(self) -> None:
        self._handle = _kernel32.CreateJobObjectW(None, None)
        if not self._handle:
            raise _error("CreateJobObjectW failed")
        self._completion_port = None
        try:
            self._completion_port = _kernel32.CreateIoCompletionPort(
                wintypes.HANDLE(INVALID_HANDLE_VALUE), None, 0, 1
            )
            if not self._completion_port:
                raise _error("CreateIoCompletionPort failed")
            association = _AssociateCompletionPort(
                CompletionKey=id(self),
                CompletionPort=self._completion_port,
            )
            if not _kernel32.SetInformationJobObject(
                self._handle,
                JOB_OBJECT_ASSOCIATE_COMPLETION_PORT_INFORMATION,
                ctypes.byref(association),
                ctypes.sizeof(association),
            ):
                raise _error("associating the job completion port failed")
            limits = _ExtendedLimitInformation()
            limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            if not _kernel32.SetInformationJobObject(
                self._handle,
                JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
                ctypes.byref(limits),
                ctypes.sizeof(limits),
            ):
                raise _error("setting job limits failed")
        except BaseException as error:
            try:
                self.close()
            except WindowsJobError as cleanup_error:
                raise WindowsJobError(f"{error}; {cleanup_error}") from error
            raise

    def assign(self, process_id: int) -> None:
        """Assign a suspended root before it can create descendants."""
        process = _kernel32.OpenProcess(PROCESS_ACCESS, False, process_id)
        if not process:
            raise _error(f"OpenProcess failed for PID {process_id}")
        with _owned_handle(
            process, f"closing process handle for PID {process_id} failed"
        ):
            if not _kernel32.AssignProcessToJobObject(self._handle, process):
                raise _error(f"AssignProcessToJobObject failed for PID {process_id}")

    def resume(self, process_id: int) -> None:
        """Resume every initial thread after job assignment."""
        snapshot = _kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0)
        if snapshot == INVALID_HANDLE_VALUE:
            raise _error("CreateToolhelp32Snapshot failed")
        resumed = 0
        with _owned_handle(snapshot, "closing the thread snapshot failed"):
            entry = _ThreadEntry(dwSize=ctypes.sizeof(_ThreadEntry))
            present = _kernel32.Thread32First(snapshot, ctypes.byref(entry))
            while present:
                if entry.th32OwnerProcessID == process_id:
                    thread = _kernel32.OpenThread(
                        THREAD_SUSPEND_RESUME, False, entry.th32ThreadID
                    )
                    if not thread:
                        raise _error(f"OpenThread failed for PID {process_id}")
                    with _owned_handle(
                        thread, f"closing thread handle for PID {process_id} failed"
                    ):
                        if _kernel32.ResumeThread(thread) == 0xFFFFFFFF:
                            raise _error(f"ResumeThread failed for PID {process_id}")
                        resumed += 1
                present = _kernel32.Thread32Next(snapshot, ctypes.byref(entry))
        if resumed == 0:
            raise WindowsJobError(f"no suspended thread found for PID {process_id}")

    def active_processes(self) -> int:
        """Return the kernel-reported number of live processes in the job."""
        if not self._handle:
            raise WindowsJobError("process job is already closed")
        information = _BasicAccountingInformation()
        if not _kernel32.QueryInformationJobObject(
            self._handle,
            JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION,
            ctypes.byref(information),
            ctypes.sizeof(information),
            None,
        ):
            raise _error("QueryInformationJobObject failed")
        return information.ActiveProcesses

    def terminate(self, timeout_seconds: float) -> None:
        """Terminate remaining processes and prove job emptiness before deadline."""
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise WindowsJobError("process-job cleanup deadline must be positive and finite")
        if not self._handle or not self._completion_port:
            raise WindowsJobError("process job is already closed")
        deadline = time.monotonic() + timeout_seconds
        if self.active_processes() == 0:
            return
        if not _kernel32.TerminateJobObject(self._handle, 1):
            code = ctypes.get_last_error()
            if self.active_processes() == 0:
                return
            raise _error_code("TerminateJobObject failed", code)

        while self.active_processes() != 0:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise WindowsJobError(
                    "process job remained active after its cleanup deadline"
                )
            message = wintypes.DWORD()
            completion_key = ctypes.c_size_t()
            overlapped = ctypes.c_void_p()
            wait_milliseconds = min(0xFFFFFFFE, max(1, math.ceil(remaining * 1000)))
            received = _kernel32.GetQueuedCompletionStatus(
                self._completion_port,
                ctypes.byref(message),
                ctypes.byref(completion_key),
                ctypes.byref(overlapped),
                wait_milliseconds,
            )
            if not received:
                code = ctypes.get_last_error()
                if code != WAIT_TIMEOUT:
                    raise _error_code("GetQueuedCompletionStatus failed", code)
            elif message.value != JOB_OBJECT_MSG_ACTIVE_PROCESS_ZERO:
                continue
            # Completion messages are advisory. QueryInformationJobObject is
            # the authoritative check, including when the zero message is lost.

    def close(self) -> None:
        """Close both handles; kill-on-close remains the last-resort boundary."""
        completion_port = self._completion_port
        handle = self._handle
        errors = []
        if handle:
            try:
                _close_handle(handle, "closing the process job failed")
            except WindowsJobError as error:
                errors.append(error)
            else:
                self._handle = None
        if completion_port:
            try:
                _close_handle(completion_port, "closing the process completion port failed")
            except WindowsJobError as error:
                errors.append(error)
            else:
                self._completion_port = None
        if errors:
            raise WindowsJobError("; ".join(map(str, errors)))
