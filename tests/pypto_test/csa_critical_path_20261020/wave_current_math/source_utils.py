# SPDX-License-Identifier: Apache-2.0
"""测试侧源码拼接辅助：只重建签名，数学体保留原中文注释。"""

import ast
import copy
import textwrap


def function(source, name):
    return next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==name)


def block(source, nodes):
    lines=source.splitlines(keepends=True)
    return textwrap.dedent("".join(lines[nodes[0].lineno-1:nodes[-1].end_lineno]))


def signature(source, name):
    fn=function(source,name)
    defaults=[None]*(len(fn.args.args)-len(fn.args.defaults))+fn.args.defaults
    return [(a.arg,ast.unparse(a.annotation),None if d is None else ast.unparse(d))
            for a,d in zip(fn.args.args,defaults)]


def header(name, arguments, returns=None):
    lines=["def "+name+"(\n"]
    for name,annotation,default in arguments:
        lines.append("    "+name+": "+annotation+(" = "+default if default is not None else "")+",\n")
    lines.append(")"+(" -> "+returns if returns else "")+":\n")
    return "".join(lines)


def replace_function(source,name,replacement):
    node=function(source,name)
    lines=source.splitlines(keepends=True)
    value="".join(lines[:node.lineno-1])+replacement.rstrip()+"\n"+"".join(lines[node.end_lineno:])
    ast.parse(value)
    return value


def body(source,name):
    return block(source,function(source,name).body)


def remove_statements(body_source,predicate):
    # 去掉的必须是明确的完整语句，不能靠字符串删掉一半call或误改数学。
    tree=ast.parse(body_source);lines=body_source.splitlines(keepends=True)
    nodes=[n for n in ast.walk(tree) if isinstance(n,ast.stmt) and predicate(n)]
    for node in sorted(nodes,key=lambda n:n.lineno,reverse=True):
        lines[node.lineno-1:node.end_lineno]=["\n"]*(node.end_lineno-node.lineno+1)
    return "".join(lines),len(nodes)


def rewrite_calls(source,name,new_args,new_targets=None):
    """按函数名重建调用行；参数表达式由回调给出，不改变其他函数或语句。"""
    tree=ast.parse(source);lines=source.splitlines(keepends=True);edits=[]
    for stmt in ast.walk(tree):
        value=stmt.value if isinstance(stmt,(ast.Assign,ast.Expr)) else None
        if not isinstance(value,ast.Call) or not isinstance(value.func,ast.Name) or value.func.id!=name:continue
        args=new_args([ast.unparse(a) for a in value.args])
        indent=" "*stmt.col_offset
        lhs=""
        if isinstance(stmt,ast.Assign):
            lhs=(new_targets(ast.unparse(stmt.targets[0])) if new_targets else ast.unparse(stmt.targets[0]))+" = "
        code=indent+lhs+name+"(\n"+"".join(indent+"    "+a+",\n" for a in args)+indent+")\n"
        edits.append((stmt.lineno-1,stmt.end_lineno,code))
    assert edits,name
    for begin,end,code in sorted(edits,reverse=True):lines[begin:end]=[code]
    result="".join(lines);ast.parse(result)
    return result,len(edits)


def algorithm_without_doc(fn):
    node=copy.deepcopy(fn)
    if node.body and isinstance(node.body[0],ast.Expr) and isinstance(node.body[0].value,ast.Constant) and isinstance(node.body[0].value.value,str):node.body.pop(0)
    return ast.dump(node,include_attributes=False)
