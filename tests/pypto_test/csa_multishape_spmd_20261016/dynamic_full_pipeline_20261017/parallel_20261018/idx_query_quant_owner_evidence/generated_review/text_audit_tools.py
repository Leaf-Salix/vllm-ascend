"""只读文本审计工具；由已验证审阅脚本抽取，不导入设备或编译环境。"""
import re
from pathlib import Path
HERE = Path(__file__).resolve().parent

def pto_function(text, name):
    start = text.index('func.func @'+name+'(')
    next_function = re.search(r'func.func @', text[start+1:])
    end = start+1+next_function.start() if next_function else len(text)
    return text[start:end]

def normalize_pto(text):
    # 只能传入单函数，常量名/数字SSA在不同函数可重复，不能跨函数收集。
    assert len(re.findall(r'func.func @', text)) == 1
    text = re.sub(r' loc\("[^"\n]+"(?::\d+:\d+)?\)', '', text)
    constants, lines, names = {}, [], {}
    for line in text.splitlines():
        match = re.fullmatch(r'\s*(%[\w.]+) = arith.constant (.+)', line)
        if match:
            constants[match[1]] = '['+match[2]+']'
        else:
            lines.append(line)
    def rename(match):
        name = match.group()
        if name in constants:
            return constants[name]
        if name not in names:
            names[name] = '%v'+str(len(names))
        return names[name]
    return re.sub(r'%[\w.]+', rename, '\n'.join(lines))

def allocations(text):
    pattern = re.compile(r'%([\w]+) = pto.alloc_tile addr = %c(\d+)_i64.*?!pto.tile_buf<'
                         r'loc=(\w+), dtype=(\w+), rows=(\d+), cols=(\d+)')
    widths = {'i8':1,'i16':2,'i32':4,'ui8':1,'ui16':2,'ui32':4,'f16':2,'bf16':2,'f32':4}
    result = []
    for match in pattern.finditer(text):
        name, address, space, dtype, rows, cols = match.groups()
        result.append(dict(name=name,address=int(address),space=space,dtype=dtype,
                           rows=int(rows),cols=int(cols),size=int(rows)*int(cols)*widths[dtype]))
    return result

def cpp_function(text, name):
    start = text.index('static __aicore__ void '+name+'(')
    brace = text.index('{',start)
    level = 1
    for i in range(brace+1,len(text)):
        if text[i] == '{': level += 1
        elif text[i] == '}': level -= 1
        if level == 0:
            return text[start:i+1]
    raise ValueError('C++函数括号不完整')

def excerpt(path, begin, end, name):
    text = path.read_text()
    first, last = text[:begin].count('\n')+1, text[:end].count('\n')+1
    lines = text.splitlines()
    (HERE/name).write_text(f'原始路径：{path}\n原始行：{first}–{last}\n\n'+
                          '\n'.join(f'{i}: {lines[i-1]}' for i in range(first,last+1))+'\n')
    return dict(path=str(path),first_line=first,last_line=last,excerpt=name)
