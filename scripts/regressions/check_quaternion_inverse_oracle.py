"""Compare actual quaternion inverses with original-component exact rationals."""
import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import random
import struct
import subprocess
parser=argparse.ArgumentParser();parser.add_argument('--base',required=True);parser.add_argument('--head',required=True);parser.add_argument('--out',required=True);args=parser.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
def decode(bits, width):
    return struct.unpack('>f' if width == 32 else '>d', bits.to_bytes(width // 8, 'big'))[0]


def encode(value, width):
    return int.from_bytes(struct.pack('>f' if width == 32 else '>d', value), 'big')


def power(exponent):
    return Fraction(1 << exponent) if exponent >= 0 else Fraction(1, 1 << -exponent)


def rounded_bits(value, width):
    """Round an exact rational once, nearest/ties-to-even, without float casts."""
    fraction_bits, bias = (23, 127) if width == 32 else (52, 1023)
    sign = (1 << (width - 1)) if value < 0 else 0
    value = abs(value)
    if not value:
        return sign
    exponent = value.numerator.bit_length() - value.denominator.bit_length()
    if value < power(exponent):
        exponent -= 1
    exponent = max(exponent, 1 - bias)
    scaled = value / power(exponent - fraction_bits)
    coefficient, remainder = divmod(scaled.numerator, scaled.denominator)
    if 2 * remainder > scaled.denominator or (
            2 * remainder == scaled.denominator and coefficient % 2):
        coefficient += 1
    if coefficient == 1 << (fraction_bits + 1):
        coefficient >>= 1
        exponent += 1
    if coefficient < 1 << fraction_bits:
        return sign | coefficient
    exponent_field = exponent + bias
    if exponent_field >= 2 * bias + 1:
        return sign | ((2 * bias + 1) << fraction_bits)
    return sign | (exponent_field << fraction_bits) | (coefficient - (1 << fraction_bits))


fixtures=[]
def add(width,label,bits):fixtures.append((width,label,bits))
for width,p,bias,maxbits in [(32,23,127,0x7f7fffff),(64,52,1023,0x7fefffffffffffff)]:
 sign=1<<(width-1);exponent_mask=(2*bias+1)<<p;maxvalue=Fraction(decode(maxbits,width));rng=random.Random(460100+width)
 # Every native power exponent, both signs and all four quaternion directions.
 for exponent in range(1-bias-p,bias+1):
  value=rounded_bits(power(exponent),width)
  assert rounded_bits(Fraction(decode(value,width)),width)==value
  for axis in range(4):
   for negative in [0,sign]:
    q=[0]*4;q[axis]=value|negative;add(width,'basis-power',q)
 pivots=[1-bias-p,1-bias,(1-bias)//2-2,(1-bias)//2-1,(1-bias)//2,(1-bias)//2+1,-2,-1,0,1,bias//2-1,bias//2,bias//2+1,bias-1,bias]
 for exponent in pivots:
  bits=rounded_bits(power(exponent),width)
  for delta in [-2,-1,0,1,2]:
   main=max(1,min(maxbits,bits+delta));v=Fraction(decode(main,width))
   for pattern in [[main]*4,[main,rounded_bits(v/2,width),rounded_bits(v/4,width),rounded_bits(v/8,width)],[main,1,0,sign],[main,max(1,main//2),max(1,main//3),max(1,main//5)]]:
    for rotation in range(4):
     q=pattern[rotation:]+pattern[:rotation];q=[b|(sign if i%2 else 0)for i,b in enumerate(q)];add(width,'exponent-boundary-mixed',q)
 for _ in range(4096):
  q=[]
  for i in range(4):
   bits=rng.getrandbits(width);q.append((bits&sign)|min(bits&(sign-1),maxbits))
  add(width,'full-bit-mixed',q)
 # Dense subnormal magnitudes around the point at which the largest inverse
 # component reaches MAX. Integer roots locate the exact norm boundary.
 count=0;eta=power(1-bias-p)
 for _ in range(10000):
  maximum=rng.randrange(1<<(p-4),1<<(p-2));a=rng.randrange(maximum+1);b=rng.randrange(maximum+1)
  target=Fraction(maximum,1)/(maxvalue*eta);remaining=target.numerator//target.denominator-maximum*maximum-a*a-b*b
  if remaining<0 or remaining>maximum*maximum:continue
  c=math.isqrt(remaining)
  for delta in [-2,-1,0,1,2]:
   q=[maximum,a,b,max(0,min(maximum,c+delta))];q=[x|(sign if rng.randrange(2)else 0)for x in q];rng.shuffle(q);add(width,'near-inverse-range-boundary',q)
  count+=1
  if count==512:break
 assert count==512
 for zeros in range(16):add(width,'signed-zero',[(sign if zeros&(1<<i)else 0)for i in range(4)])
 specials=[exponent_mask,exponent_mask|sign,exponent_mask|1,exponent_mask|(1<<(p-1))|0x123,exponent_mask|sign|1,exponent_mask|sign|(1<<(p-1))|0x123]
 for special in specials:
  for axis in range(4):
   for context in [[encode(1.0,width)]*4,[maxbits]*4,[1]*4,[0]*4]:
    q=context.copy();q[axis]=special;add(width,'nonfinite-observation',q)
text=''.join(str(w)+' '+ ' '.join(format(b,'x')for b in q)+'\n'for w,_,q in fixtures);(out/'inputs.txt').write_text(text)
def run(exe,label):
 r=subprocess.run([exe],input=text,text=True,capture_output=True,check=True,timeout=60);(out/(label+'.tsv')).write_text(r.stdout);rows=[]
 for i,line in enumerate(r.stdout.splitlines()):
  f=line.split();assert int(f[0])==i;rows.append((int(f[1]),None if f[2]=='None'else tuple(map(int,f[3:7])),None if f[2]=='None'else tuple(map(int,f[7:]))))
 assert len(rows)==len(fixtures);return rows
base=run(args.base,'base');head=run(args.head,'head');counts={w:{'cases':0,'nonfinite_preserved':0,'zero_preserved':0,'ordinary_native_preserved':0,'finite_inverse_cases':0,'repaired_missing':0,'repaired_values':0,'excluded_unrepresentable_inverse':0,'max_fallback_ulp':0,'max_ordinary_ulp':0,'both_orders_checked':0,'max_product_residual_eps':0}for w in [32,64]};errors=[]
for (width,label,bits),(base_norm,b,base_products),(head_norm,h,products)in zip(fixtures,base,head):
 c=counts[width];c['cases']+=1;assert base_norm==head_norm
 p,bias=(23,127)if width==32 else(52,1023);sign=1<<(width-1);mask=(2*bias+1)<<p
 values=[decode(v,width)for v in bits]
 if any(not math.isfinite(v)for v in values):
  if b!=h:errors.append([width,label,'nonfinite_changed',bits,b,h])
  c['nonfinite_preserved']+=1;continue
 if not any(values):
  if b is not None or h is not None:errors.append([width,label,'zero_changed',bits,b,h])
  c['zero_preserved']+=1;continue
 values=list(map(Fraction,values));norm=sum(v*v for v in values);inverse=[(v if i==0 else-v)/norm for i,v in enumerate(values)];expected=[rounded_bits(v,width)for v in inverse]
 if any(v&mask==mask for v in expected):
  c['excluded_unrepresentable_inverse']+=1;continue
 c['finite_inverse_cases']+=1
 ordinary=(base_norm&mask)!=0 and(base_norm&mask)!=mask
 if ordinary:
  if b!=h:errors.append([width,label,'ordinary_bits_changed',bits,b,h])
  c['ordinary_native_preserved']+=1
 if h is None or any(v&mask==mask for v in h):errors.append([width,label,'lost_finite_inverse',bits,expected,b,h]);continue
 c['both_orders_checked']+=1
 eps=power(-p)
 residual=max(abs(Fraction(decode(v,width))-(1 if i%4==0 else 0))for i,v in enumerate(products))
 c['max_product_residual_eps']=max(c['max_product_residual_eps'],float(residual/eps))
 if residual>16*eps:errors.append([width,label,'identity_product_residual',bits,h,products])
 if b is None:c['repaired_missing']+=1
 elif b!=h:c['repaired_values']+=1
 for actual,expected_component in zip(h,expected):
  # Signed zeros denote the same exact value; differential preservation above
  # separately checks all ordinary and nonfinite result bits.
  distance=abs((actual&(sign-1))-(expected_component&(sign-1)))if(actual&sign)==(expected_component&sign)or not(actual&(sign-1))and not(expected_component&(sign-1))else 1<<width
  key='max_ordinary_ulp'if ordinary else'max_fallback_ulp';c[key]=max(c[key],distance)
  if distance>(6 if ordinary else 3):errors.append([width,label,'ulp_bound',distance,bits,expected,h])
receipt={'status':'PASS'if not errors else'FAIL','cases':len(fixtures),'counts':counts,'errors':errors[:30],'total_errors':len(errors),'corpus_sha256':hashlib.sha256(text.encode()).hexdigest(),'oracle':'Exact Fraction norm and inverse from original IEEE components, explicit integer nearest/ties-even IEEE rounding.','product_bound':'Both native Hamilton multiplication orders stay within16epsilon absolute of identity. Component inverse errors plus four nonnegative magnitude weights and seven-operation product rounding are bounded separately; tiny inverse outputs add absolute subnormal terms.', 'bounds':'Ordinary unchanged native path:6ulp (positive four-term squared norm and one division). Compensated scaled fallback:3ulp incl final subnormal rounding. Neither is a universal correctly-rounded promise.','unrepresentable_scope':'Inverse components whose exact nearest native value is infinity are counted separately, not certified as finite results or a new range policy.'};(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2));assert not errors
