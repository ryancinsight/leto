#!/usr/bin/env python3
"""Check public fixed-vector/Unit results against a high-precision direction oracle."""
from pathlib import Path
import argparse,decimal,hashlib,json,math,random,struct,subprocess
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--base',type=Path,required=True)
parser.add_argument('--head',type=Path,required=True)
parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args();out=args.out;out.mkdir(parents=True,exist_ok=True)
decimal.getcontext().prec=100;D=decimal.Decimal;inputs=[];rng=random.Random(1071702)
def value(w,b):return struct.unpack('>f'if w==32 else'>d',b.to_bytes(w//8,'big'))[0]
def bits(w,v):return int.from_bytes(struct.pack('>f'if w==32 else'>d',v),'big')
for w,p,bias,lo,hi in [(32,23,127,-149,127),(64,52,1023,-1074,1023)]:
 sign=1<<(w-1);inf=((1<<(w-p-1))-1)<<p;maximum=inf-1
 for n in [0,1,2,3,4,8,17]:
  rows=[]
  if n:
   for special in [0,1,(1<<p)-1,1<<p,maximum,inf,inf|sign,inf|1,inf|(1<<(p-1))]:
    rows.extend([[special]+[0]*(n-1),[special]*n])
   for e in range(lo,hi+1):
    a=bits(w,math.ldexp(1.0,e));rows.extend([[a]+[0]*(n-1),[a^(sign if i%2 else 0)for i in range(n)],[a if i==0 else (1 if i%2 else 0)for i in range(n)]])
   rows.extend([[rng.getrandbits(w)for _ in range(n)]for _ in range(80)])
  else:rows=[[]]
  for row in dict.fromkeys(tuple(x)for x in rows):inputs.append((w,n,row))
data=''.join(f'{w} {n} '+ ' '.join(f'{b:x}'for b in row)+'\n'for w,n,row in inputs).encode();(out/'inputs.txt').write_bytes(data)
references=[]
for w,n,row in inputs:
 vals=[value(w,b)for b in row];finite=all(math.isfinite(x)for x in vals);nonzero=any(x!=0 for x in vals)
 if finite and nonzero:
  exact=[D.from_float(x)for x in vals];norm=sum(x*x for x in exact).sqrt();references.append([x/norm for x in exact])
 else:references.append(None)
outputs={}
for label,exe in [('base',args.base),('repair',args.head)]:
 p=subprocess.run([str(exe.resolve())],input=data,capture_output=True,timeout=120)
 if p.returncode:raise RuntimeError(p.stderr.decode())
 (out/(label+'.tsv')).write_bytes(p.stdout)
 outputs[label]=[line.split()for line in p.stdout.decode().splitlines()]
opt='public-executable'
assert all(len(x)==len(inputs)*2 for x in outputs.values())
ordinary=finite_count=invalid=0;max_component_eps=0.;max_norm_eps=0.
for i,((w,n,row),reference)in enumerate(zip(inputs,references)):
 old,oldwrap=outputs['base'][i*2:i*2+2];new,wrap=outputs['repair'][i*2:i*2+2];assert new[:1]+new[2:]==wrap[:1]+wrap[2:],(opt,i,'wrapper mismatch')
 p=23 if w==32 else 52;eps=D(2)**(-p);mask=((1<<(w-p-1))-1)<<p;norm_bits=int(new[2]);assert old[2]==new[2]or(math.isnan(value(w,int(old[2])))and math.isnan(value(w,norm_bits)))
 actual=[value(w,int(x))for x in new[3:3+n]];flag=new[3+n]
 if reference is not None:
  finite_count+=1;assert flag=='Some'and all(math.isfinite(x)for x in actual),(opt,i,new)
  assert new[3:3+n]==new[4+n:],(opt,i,'checked bits')
  error=max((abs(D.from_float(a)-b)for a,b in zip(actual,reference)),default=D(0));assert error<=eps*(n+8),(opt,i,'direction',error)
  residual=abs(sum(D.from_float(x)**2 for x in actual)-1);assert residual<=eps*(2*n+16),(opt,i,'norm',residual)
  max_component_eps=max(max_component_eps,float(error/eps));max_norm_eps=max(max_norm_eps,float(residual/eps))
  if norm_bits&mask and norm_bits&mask!=mask:ordinary+=1;assert old==new,(opt,i,'ordinary bits changed',old,new)
 else:
  invalid+=1;assert flag=='None',(opt,i,'invalid accepted')
  for a,b in zip(old[3:3+n],new[3:3+n]):assert a==b or(math.isnan(value(w,int(a)))and math.isnan(value(w,int(b)))),(opt,i,'legacy IEEE')
receipt={'status':'PASS','cases':len(inputs),'input_sha256':hashlib.sha256(data).hexdigest(),'finite_nonzero_cases':finite_count,'zero_nonfinite_empty':invalid,'ordinary_bits_preserved':ordinary,'wrapper_bits_preserved_all':True,'max_component_error_eps':max_component_eps,'max_squared_norm_residual_eps':max_norm_eps,'dimensions':[0,1,2,3,4,8,17],'scalar_types':['f32','f64','F32','F64'],'oracle':'Decimal100-digit norm/direction from exact original IEEE components. For tested N, positive norm summation and sqrt/divide rounding permit(N+8)epsilon absolute component and(2N+16)epsilon squared-unit residual bounds, including subnormal terms. This is not a universal correctly-rounded guarantee.','legacy_nonfinite':'Direct zero/nonfinite results retain IEEE classifications. Checked zero/nonfinite/empty inputs reject.','ordinary':'Finite normal scalar norms preserve the original reciprocal-multiply output bits.'}
(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
