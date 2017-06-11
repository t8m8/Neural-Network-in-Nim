import macros

proc toNewCall(node: NimNode): NimNode {.compiletime.} =
  case node.kind
  of nnkBracketExpr:
    let ident = $node[0]
    result = newCall("new" & ident, node[1], node[2])
  of nnkCall:
    let ident = $node[0]
    result = newCall("new" & ident, node[1])
  of nnkIdent:
    let ident = $node
    result = newCall("new" & ident)
  else:
    discard

proc layers(body, builder: NimNode, stmtlist: var NimNode) {.compiletime.} =
  let
    addExpr = newDotExpr(builder, newIdentNode("add"))
  body.expectKind(nnkStmtList)
  for `stmt` in body:
    stmtlist.add newAssignment(builder, newCall(addExpr, toNewCall(`stmt`)))

proc minimize(body: NimNode, builder, stmtlist: var NimNode) {.compiletime.} =
  let minimizeExpr = newDotExpr(builder, newIdentNode("minimize"))
  var builderWithLoss = newIdentNode("innerBuilderWithLoss")
  stmtlist.add newVarStmt(builderWithLoss, newCall(minimizeExpr, toNewCall(body[0])))
  builder = builderWithLoss

proc optimize(body: NimNode, builder, stmtlist: var NimNode) {.compiletime.} =
  let optimizeExpr = newDotExpr(builder, newIdentNode("optimize"))
  var builderWithOptimize = newIdentNode("innerBuilderWithOptimize")
  stmtlist.add newVarStmt(builderWithOptimize, newCall(optimizeExpr, toNewCall(body[0])))
  var network = newIdentNode("network")
  stmtlist.add(quote do:
    var `network` {.used.} = `builderWithOptimize`.build()
  )
  builder = network

macro network*(body: untyped): untyped =
  body.expectKind(nnkStmtList)
  var
    builder = newIdentNode("innerBuilder")
    stmtlist = newStmtList()
  stmtlist.add(quote do:
    var `builder`: NNBuilder = newNNBuilder()
  )

  for `block` in body:
    case `block`.kind
    of nnkCall:
      if `block`[0].kind != nnkIdent:
        stmtlist.add `block`
      elif $`block`[0] == "layers":
        layers(`block`[1], builder, stmtlist)
      elif $`block`[0] == "minimize":
        minimize(`block`[1], builder, stmtlist)
      elif $`block`[0] == "optimize":
        optimize(`block`[1], builder, stmtlist)
      else:
        stmtlist.add `block`
    else:
      stmtlist.add `block`

  result = newBlockStmt(stmtlist)