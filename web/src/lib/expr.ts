// Same grammar as the C++ evaluator (src/Graph/Expr.cpp): + - * / // % and
// parentheses over numbers and variables written as name or $name.

export function evaluate(text: string, vars: Record<string, number>): number {
  let i = 0;
  const s = text;
  const skip = () => {
    while (i < s.length && /\s/.test(s[i])) i++;
  };
  const accept = (t: string) => {
    skip();
    if (s.startsWith(t, i) && !(t === '/' && s.startsWith('//', i))) {
      i += t.length;
      return true;
    }
    return false;
  };
  const expr = (): number => {
    let v = term();
    for (;;) {
      if (accept('+')) v += term();
      else if (accept('-')) v -= term();
      else return v;
    }
  };
  const term = (): number => {
    let v = factor();
    for (;;) {
      if (accept('//')) v = Math.floor(v / nonzero(factor()));
      else if (accept('*')) v *= factor();
      else if (accept('/')) v /= nonzero(factor());
      else if (accept('%')) v %= nonzero(factor());
      else return v;
    }
  };
  const nonzero = (v: number) => {
    if (v === 0) throw new Error('division by zero');
    return v;
  };
  const factor = (): number => {
    if (accept('-')) return -factor();
    if (accept('+')) return factor();
    skip();
    if (accept('(')) {
      const v = expr();
      if (!accept(')')) throw new Error("missing ')'");
      return v;
    }
    const num = /^(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?/.exec(s.slice(i));
    if (num) {
      i += num[0].length;
      return Number(num[0]);
    }
    if (s[i] === '$') i++;
    const id = /^[A-Za-z_]\w*/.exec(s.slice(i));
    if (!id) throw new Error('expected a number or variable');
    i += id[0].length;
    if (!(id[0] in vars)) throw new Error(`unknown variable '${id[0]}'`);
    return vars[id[0]];
  };
  const v = expr();
  skip();
  if (i !== s.length) throw new Error(`unexpected '${s[i]}'`);
  return v;
}
