"""Windows Job Object ownership for bounded subprocess trees."""

from __future__ import annotations

import ctypes
import math
from ctypes import wintypes

CREATE_SUSPENDED = 0x00000004
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
PROCESS_ACCESS = 0x00100101
THREAD_SUSPEND_RESUME = 0x0002
TH32CS_SNAPTHREAD = 0x00000004
WAIT_OBJECT_0 = 0
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
_kernel32.WaitForSingleObject.restype = wintypes.DWORD
_kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
_kernel32.CloseHandle.restype = wintypes.BOOL
_kernel32.CloseHandle.argtypes = [wintypes.HANDLE]


def _error(action: str) -> WindowsJobError:
    return WindowsJobError(f"{action}: Windows error {ctypes.get_last_error()}")


def _error_code(action: str, code: int) -> WindowsJobError:
    return WindowsJobError(f"{action}: Windows error {code}")


class WindowsJob:
    """Own one suspended root process and all descendants in one OS job."""

    def __init__(self) -> None:
        self._handle = _kernel32.CreateJobObjectW(None, None)
        if not self._handle:
            raise _error("CreateJobObjectW failed")
        limits = _ExtendedLimitInformation()
        limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not _kernel32.SetInformationJobObject(
            self._handle,
            JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(limits),
            ctypes.sizeof(limits),
        ):
            code = ctypes.get_last_error()
            self.close()
            raise _error_code("SetInformationJobObject failed", code)

    def assign(self, process_id: int) -> None:
        """Assign a suspended root before it can create descendants."""
        process = _kernel32.OpenProcess(PROCESS_ACCESS, False, process_id)
        if not process:
            raise _error(f"OpenProcess failed for PID {process_id}")
        try:
            if not _kernel32.AssignProcessToJobObject(self._handle, process):
                raise _error(f"AssignProcessToJobObject failed for PID {process_id}")
        finally:
            _kernel32.CloseHandle(process)

    def resume(self, process_id: int) -> None:
        """Resume every initial thread after job assignment."""
        snapshot = _kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0)
        if snapshot == INVALID_HANDLE_VALUE:
            raise _error("CreateToolhelp32Snapshot failed")
        resumed = 0
        entry = _ThreadEntry(dwSize=ctypes.sizeof(_ThreadEntry))
        try:
            present = _kernel32.Thread32First(snapshot, ctypes.byref(entry))
            while present:
                if entry.th32OwnerProcessID == process_id:
                    thread = _kernel32.OpenThread(
                        THREAD_SUSPEND_RESUME, False, entry.th32ThreadID
                    )
                    if not thread:
                        raise _error(f"OpenThread failed for PID {process_id}")
                    try:
                        if _kernel32.ResumeThread(thread) == 0xFFFFFFFF:
                            raise _error(f"ResumeThread failed for PID {process_id}")
                        resumed += 1
                    finally:
                        _kernel32.CloseHandle(thread)
                present = _kernel32.Thread32Next(snapshot, ctypes.byref(entry))
        finally:
            _kernel32.CloseHandle(snapshot)
        if resumed == 0:
            raise WindowsJobError(f"no suspended thread found for PID {process_id}")

    def terminate(self, timeout_seconds: float) -> None:
        """Terminate the entire job and wait within the supplied bound."""
        if not self._handle:
            raise WindowsJobError("process job is already closed")
        if not _kernel32.TerminateJobObject(self._handle, 1):
            raise _error("TerminateJobObject failed")
        milliseconds = max(1, math.ceil(timeout_seconds * 1000))
        result = _kernel32.WaitForSingleObject(self._handle, milliseconds)
        if result == WAIT_TIMEOUT:
            raise WindowsJobError("process job did not terminate before its deadline")
        if result != WAIT_OBJECT_0:
            raise _error("WaitForSingleObject failed")

    def close(self) -> None:
        """Close the job; kill-on-close remains the last-resort boundary."""
        if self._handle:
            _kernel32.CloseHandle(self._handle)
            self._handle = None
