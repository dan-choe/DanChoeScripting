# -----------------------------------------------------------------------------
# danChoeScript.py
#
# Dan Choe
# Nov 23. 2017
# -----------------------------------------------------------------------------

import ply.lex as lex
import sys

class Node:
    def __init__(self):
        print("init node")

    def evaluate(self) -> object:
        return 0

    def execute(self):
        return 0


class ArrayNode(Node):
    def __init__(self, v):
        self.value = v

    def evaluate(self) -> object:
        return self.value


class BlockNode(Node):
    def __init__(self, parent):
        self.childNodes = []
        self.parent = parent
        self.singleStatement = False
        self.type = 'normal'

    # Add statements and sub-blocks
    def addStm(self, s):
        # if type(s) == type(''):
        #     s.replace('}', '')
        #     if s == '':
        #         return
        self.childNodes.append(s)

    # if/else does not have curlybraces
    def setSingleStm(self, status):
        self.singleStatement = status



class FunctionNode(Node):
    def __init__(self, parent):
        self.childNodes = []
        self.parent = parent
        self.returnValue = None
        self.localVariables = {}
        self.type = 'func'
        self.argumentOrder = []

    def setArgument(self, argName, argV):
        self.argumentOrder.append(argName)
        self.localVariables[argName] = argV

    # Add statements and sub-blocks
    def addStm(self, s):
        self.childNodes.append(s)

    # For returnvalue of funtions
    def setReturn(self, v):
        self.returnValue = v

    def addLocalVariable(self, item, v):
        self.localVariables[str(item)] = v

    # return ReturnValue
    def evaluate(self) -> object:
        return self.returnValue

    def execute(self):
        global currRunModeFunction
        global currRunMode

        # print('\nFunctionNode execute = run')

        currRunModeFunction = self
        currRunMode = 1

        tempNode = BlockNode(None)
        tempNode.childNodes = self.childNodes[1:]

        recurseBlocks(tempNode)

        # currRunModeFunction = None
        # currRunMode = 0
        return self.returnValue




class VariableNode(Node):
    def __init__(self, name, v):
        self.name = str(name)
        self.value = v

    def evaluate(self) -> object:
        return self.value


class StringNode(Node):
    def __init__(self, v):
        self.value = str(v)

    def evaluate(self) -> object:
        return self.value


class NumberNode(Node):
    def __init__(self, v):
        if('.' in v):
            self.value = float(v)
        else:
            self.value = int(v)

    def evaluate(self) -> object:
        return self.value


class PrintNode(Node):
    def __init__(self, v):
        # print('printnode made')
        self.value = v

    def execute(self):
        # print('\n\n\n\n [[[[[[[[ printNode ]]]]]]]]]]] ')

        if(type(self.value) != type(1) and type(self.value) != type('') and type(self.value) != type([])):
            self.value = self.value.evaluate()
        # print('temp [printFunction] : ', self.value)

        if (type(self.value) == type(1)):
            # print('[printFunction primary - number] : ', self.value)
            print(self.value)
        elif (type(self.value) == type('')):
            temp = '\'' + self.value + '\''
            # print('[printFunction primary - string] : ', temp)
            print(temp)
        elif (type(self.value) == type([])):
            temp = []
            for i in self.value:
                if (type(i) == type(1)):
                    temp.append(i)
                elif (type(i) == type('')):
                    temp.append(i)
                elif (type(i) == type([])):
                    temp_sub = []
                    temp.append(temp_sub)
                    for j in i:
                        if (type(j) == type(1) or type(j) == type('') or type(j) == type([])):
                            temp_sub.append(j)
                        else:
                            temp_sub.append(j.value)
                else:
                    temp.append(i.value)
            # print('[printFunction] : ', temp)
            # print('---begin')
            print(temp)
            # print('---end')
        else:
            # print('[printFunction evaluate] : ', self.evaluate())
            print(self.evaluate())


class BopNode(Node):
    def __init__(self, op, v1, v2):
        self.v1 = v1
        self.v2 = v2
        self.op = op

    def evaluate(self) -> object:
        try:

            # print('bopnode : ', self.v1, self.op, self.v2)

            if (self.op == '+'):
                return self.v1.evaluate() + self.v2.evaluate()
            elif (self.op == '-'):
                return self.v1.evaluate() - self.v2.evaluate()
            elif (self.op == '**'):
                return self.v1.evaluate() ** self.v2.evaluate()
            elif (self.op == '*'):
                return self.v1.evaluate() * self.v2.evaluate()
            elif (self.op == '//'):
                return self.v1.evaluate() // self.v2.evaluate()
            elif (self.op == '/'):
                return self.v1.evaluate() / self.v2.evaluate()
            elif (self.op == '%'):
                return self.v1.evaluate() % self.v2.evaluate()
            elif (self.op == 'or'):
                return self.v1.evaluate() or self.v2.evaluate()
            elif (self.op == 'and'):
                return self.v1.evaluate() and self.v2.evaluate()
            elif (self.op == 'in'):
                return self.v1.evaluate() in self.v2.evaluate()
        except AttributeError:
            global result
            result = 'SYNTAX ERROR'

class CompareNode(Node):
    def __init__(self, op, v1, v2):
        self.v1 = v1
        self.v2 = v2
        self.op = op

    def evaluate(self) -> object:
        global result
        try:

            # print('\n Acutal CompareNode : ', self.v1, self.op, self.v2)


            if (self.op == '=='):

                # print('self.v1.evaluate() ', self.v1.evaluate())
                #
                # print('self.v2.evaluate() ', self.v2.evaluate())

                return self.v1.evaluate() == self.v2.evaluate()
            elif (self.op == '<='):
                return self.v1.evaluate() <= self.v2.evaluate()
            elif (self.op == '>='):
                return self.v1.evaluate() >= self.v2.evaluate()
            elif (self.op == '<>'):
                return self.v1.evaluate() != self.v2.evaluate()
            elif (self.op == '<'):
                return self.v1.evaluate() < self.v2.evaluate()
            elif (self.op == '>'):
                return self.v1.evaluate() > self.v2.evaluate()
        except AttributeError:
            result = 'SYNTAX ERROR'
            print('CompareNode error: ')


tokens = (
    'PRINT','EQUAL','ASSIGN','VARNAME','SEMICOLON', 'IF', 'ELSE', 'WHILE','RETURN',
    'NUMBER','REAL','STRING','COMMA','LCURLYBRACE','RCURLYBRACE',
    'PLUS','MINUS','POWER','TIMES','DIVIDE2','DIVIDE','MODULUS',
    'LPAREN','RPAREN','SQOPEN','SQCLOSE','IN','OR','AND','NOT',
    'LESSE','GREATE','UNEQUAL','LESS','GREAT'
    )

# Tokens
t_EQUAL = r'(\==)|(\==\s)|(\s\==)'

t_SEMICOLON = r'(\;)|(\;\s)|(\s\;)'

t_LCURLYBRACE = r'({)|({\s)|(\s{)'
t_RCURLYBRACE = r'(})|(}\s)|(\s})'


t_LESSE  = r'<='
t_GREATE  = r'>='
t_UNEQUAL = r'<>'
t_LESS  = r'<'
t_GREAT  = r'>'

t_ASSIGN = r'(=)|(=\s)|(\s=)'

t_PLUS    = r'\+'
t_MINUS   = r'-'
t_POWER  = r'(\*\*)'
t_TIMES   = r'\*'
t_DIVIDE2  = r'//'
t_DIVIDE  = r'/'
t_MODULUS  = r'%'

t_LPAREN  = r'\('
t_RPAREN  = r'\)'
t_SQOPEN  = r'\['
t_SQCLOSE = r'\]'
t_COMMA = r'(\,)|(\,\s)|(\s\,)'


def t_RETURN(t):
    r'return'
    try:
        t.value = t.value
    except ValueError:
        t.value = ''
    return t


def t_IN(t):
    r'in'
    try:
        t.value = t.value
    except ValueError:
        t.value = ''
    return t

def t_OR(t):
    r'or'
    try:
        t.value = str(t.value)
    except ValueError:
        t.value = ''
    return t

def t_AND(t):
    r'and'
    try:
        t.value = str(t.value)
    except ValueError:
        t.value = ''
    return t

def t_NOT(t):
    r'not'
    try:
        t.value = str(t.value)
    except ValueError:
        t.value = ''
    return t

def t_REAL(t):
    r'[-+]?((((\.\d+)|(\d*\.\d*))([eE][-+]?\d+)?)|((\d+)([eE][-+]?\d+)))'
    try:
        t.value = NumberNode(t.value)
        #t.value = float(t.value)
    except ValueError:
        t.value = 0
    return t

def t_NUMBER(t):
    r'\d+'
    try:
        t.value = NumberNode(t.value)
        #t.value = int(t.value)
    except ValueError:
        t.value = 0
    return t


def t_STRING(t):
    r'[\"]([a-zA-Z0-9_:;\(\)\-\.\+?!\s\*&])*[\"]'
    try:
        t.value = str(t.value).replace('\"','')
        t.value = StringNode(t.value)
        # print('\n t_STRING.value : ', t.value)
    except ValueError:
        t.value = ""
    return t


def t_PRINT(t):
    r'print'
    try:
        t.value = t.value
        # t.value = VarNode(t.value).value
        # print('t_PRINT.value : ', t.value)
    except ValueError:
        t.value = ""
    return t

def t_WHILE(t):
    r'while'
    try:
        # t.value = VarNode(t.value).value
        t.value = t.value
        # print('t_IF.value : ', t.value)
    except ValueError:
        t.value = ""
    return t

def t_IF(t):
    r'if'
    try:
        # t.value = VarNode(t.value).value
        t.value = t.value
        # print('t_IF.value : ', t.value)
    except ValueError:
        t.value = ""
    return t


def t_ELSE(t):
    r'else'
    try:
        # t.value = VarNode(t.value).value
        t.value = t.value
        # print('t_ELSE.value : ', t.value)
    except ValueError:
        t.value = ""
    return t


# r'(?!\')[a-zA-Z][a-zA-Z0-9_]*(?!\')'
# r'[^\'][a-zA-Z][a-zA-Z0-9_]*[^\']'
def t_VARNAME(t):
    r'[a-zA-Z][a-zA-Z0-9_]*'
    try:
        # print('varname token: ', t.value)
        t.value = t.value
    except ValueError:
        t.value = ""
    return t

# Ignored characters
t_ignore = " \t"

def t_newline(t):
    r'\n+'
    t.lexer.lineno += t.value.count("\n")

def t_error(t):
    t.lexer.skip(1)

# Build the lexer
lexer = lex.lex()

# Parsing rules
precedence = (
    ('left','PLUS','MINUS'),
    ('left','TIMES','POWER','DIVIDE2','DIVIDE','MODULUS'),
    ('right','IN','OR','AND','NOT')
    )

# global variables
result = None

currStatus = True
ifElse = ''


variables = {}
blocks = []
functionBlocks = {}

# brace stack
braceStack = []
blockStack = []


# one line block
def p_statement_openclose(t):
    '''
        expression : LCURLYBRACE expression RCURLYBRACE
    '''
    global result

    try:
        # print('1. p_statement_openclose', type(t[2]))
        braceStack.append('{')
        blockStack.append(t[2])
        braceStack.append('}')

        t[0] = t[2]

        if(type(t[0]) == type('')): # string type
            t[0] = '\'' + t[0] + '\''
        if result != 'SYNTAX ERROR' and result != 'SEMANTIC ERROR':
            result = None

    except:
        result = 'SEMANTIC ERROR'


# open block
def p_statement_open(t):
    '''
        statement : LCURLYBRACE expression
    '''
    global result
    global braceStack
    global blockStack

    try:
        # print('1-1. p_statement_open', t[2])

        braceStack.append('{')
        blockStack.append(t[2])

        t[0] = t[2]

        if(type(t[0]) == type('')): # string type
            t[0] = '\'' + t[0] + '\''
        if result != 'SYNTAX ERROR' and result != 'SEMANTIC ERROR':
            result = None

    except:
        result = 'SEMANTIC ERROR'




def p_single_openBrace(t):
    '''
        expression : LCURLYBRACE
    '''
    global result
    global braceStack
    global blockStack
    try:
        t[0] = t[0]
        braceStack.append('{')
    except TypeError:
        result = 'SEMANTIC ERROR'


def p_single_closeBrace(t):
    '''
        expression : RCURLYBRACE
    '''
    global result
    global braceStack
    global blockStack
    try:
        t[0] = t[0]
        braceStack.append('}')
    except TypeError:
        result = 'SEMANTIC ERROR'


def p_print_TEMP(t):
    '''
        expression : PRINT LPAREN expression RPAREN
    '''
    global result
    global braceStack
    global blockStack

    try:
        # print('\n\n [debug] print call')
        # print('PRINT ', t[1])
        # print('expression ', t[3])

        if result == None:
            t[0] = PrintNode(t[3])
            t[0].execute()
    except TypeError:
        result = 'SEMANTIC ERROR'



def p_print_smt(t):
    '''
        expression : PRINT LPAREN expression RPAREN SEMICOLON
    '''
    global result
    global braceStack
    global blockStack

    try:
        # print('\n\n [debug] print call')
        # print('PRINT ', t[1])
        # print('expression ', t[3])

        if result == None:
            t[0] = PrintNode(t[3])
            t[0].execute()
    except TypeError:
        result = 'SEMANTIC ERROR'



def p_function_return(t):
    '''
        expression : RETURN expression SEMICOLON
    '''
    global result
    global braceStack
    global blockStack
    global functionBlocks
    global currRunMode
    global currRunModeFunction

    try:
        # print('\n\n[debug 2] function RETURN')


        if type(t[2]) == type(BopNode(1,1,1)):
            t[0] = t[2].evaluate()

            currRunModeFunction.returnValue = t[0]

            if type(t[0]) == type(0):
                t[0] = NumberNode(str(t[0]))
            elif type(t[0]) == type(''):
                t[0] = StringNode(str(t[0]))

            # print('expression1 ', t[2].evaluate())
        else:
            # print('--------', t[2])

            v = t[2].evaluate()

            if type(v) == type(0):
                t[0] = NumberNode(str(v))
                currRunModeFunction.returnValue = t[0].evaluate()
            elif type(v) == type(''):
                t[0] = StringNode(str(v))
                currRunModeFunction.returnValue = t[0].evaluate()
            else:
                t[0] = v
                currRunModeFunction.returnValue = t[0]
            # print('expression2 ', t[2].evaluate())
            # print('type: ', type(t[0]))
        # currRunMode = 0
        # currRunModeFunction = None

    except TypeError:
        result = 'SEMANTIC ERROR'


def p_functionCall_under(t):
    '''
        expression : VARNAME LPAREN factor RPAREN
    '''
    global result
    global braceStack
    global blockStack
    global functionBlocks
    global currRunMode
    global currRunModeFunction

    try:
        # print('\n\n[debug 2] function call  -->', t[1], t[2], t[3], t[4])
        # print('VARNAME ', t[1])
        # print('expression ', t[3])

        # find target function
        if t[1] in functionBlocks.keys():

            targetFunction = functionBlocks[t[1]]
            # print('target args: ', targetFunction.localVariables.keys())

            # print('\ttarget function. # of args : ', len(targetFunction.argumentOrder), 'parse value: ', t[3])

            # Setting Arguments to parse to the function
            if len(targetFunction.argumentOrder) > 1 and type(t[3]) == type([]):
                if len(targetFunction.argumentOrder) != len(t[3]):
                    result = 'SEMANTIC ERROR'
                else:
                    for i in range(len(t[3])):
                        targetFunction.addLocalVariable(targetFunction.argumentOrder[i], t[3][i])

            elif len(targetFunction.argumentOrder) < 2:

                if len(targetFunction.argumentOrder) == 1:
                    if type(t[3]) == type([]):
                        targetFunction.addLocalVariable(targetFunction.argumentOrder[0], t[3][0])
                    else:
                        targetFunction.addLocalVariable(targetFunction.argumentOrder[0], t[3])
            else:
                result = 'SEMANTIC ERROR'

            # No error. Run the function
            # print(' -------------------------- ', result)

            if result == None:
                currRunMode = 1
                currRunModeFunction = targetFunction

                # print('check vars: ', targetFunction.localVariables)

                t[0] = targetFunction
                t[0].execute()

        else:
            # print('the function does not exist.')
            result = 'SEMANTIC ERROR'
            t[0] = 0
    except TypeError:
        result = 'SEMANTIC ERROR'



def p_with_brace(t):
    '''
        expression : WHILE LPAREN expression RPAREN LCURLYBRACE
                    | IF LPAREN expression RPAREN LCURLYBRACE
    '''
    global result
    global braceStack
    global blockStack
    try:
        # print('>> 1   ( ) { statement: ', t[1], t[3], t[3].evaluate())
        t[0] = t[3].evaluate()
    except TypeError:
        result = 'SEMANTIC ERROR'


def p_without_brace(t):
    '''
        expression : WHILE LPAREN expression RPAREN
                    | IF LPAREN expression RPAREN
    '''
    global result
    global braceStack
    global blockStack
    try:
        # print('>>  2 statement: ', t[1], t[2], t[3], t[4])
        t[0] = t[3].evaluate()
        # print('>>  2 statement evaluate: ', t[0], t[1], t[3], t[3].evaluate())
    except TypeError:
        result = 'SEMANTIC ERROR'


def p_else_smt(t):
    '''
        expression : RCURLYBRACE ELSE LCURLYBRACE
    '''
    global result
    global braceStack
    global blockStack
    try:
        t[0] = t[0]
        braceStack.append('{')
        braceStack.append('}')
    except TypeError:
        result = 'SEMANTIC ERROR'


def p_single_else(t):
    '''
        expression : ELSE
    '''
    global result
    global braceStack
    global blockStack
    try:
        t[0] = t[0]
    except TypeError:
        result = 'SEMANTIC ERROR'




def p_varname_compare2(t):
    '''
        expression : expression EQUAL primary
              | expression LESSE primary
              | expression GREATE primary
              | expression UNEQUAL primary
              | expression LESS primary
              | expression GREAT primary
    '''
    global result
    try:
        # print('compare 2: ', t[1], t[2], t[3])

        if( type(t[1]) != type(t[3]) and ( type(t[1]) == type('') or type(t[3]) == type('') )):
            # print('compare error found: ', t[1], t[2], t[3])
            result = 'SEMANTIC ERROR'
            t[0] = t[3]
        else:
            t[0] = CompareNode(t[2], t[1], t[3])
        # print('compare generated: ', t[0], t[1], t[2], t[3])
        # print('compare result: ', t[0].evaluate())
    except TypeError:
        result = 'SEMANTIC ERROR'


def p_varname_compare(t):
    '''
        expression : primary EQUAL expression
              | primary LESSE expression
              | primary GREATE expression
              | primary UNEQUAL expression
              | primary LESS expression
              | primary GREAT expression
    '''
    global result
    try:
        # print('compare 1: ', t[1], t[2], t[3])

        if( type(t[1]) != type(t[3]) and ( type(t[1]) == type('') or type(t[3]) == type('') )):
            # print('compare error found: ', t[1], t[2], t[3])
            result = 'SEMANTIC ERROR'
            t[0] = t[3]
        else:
            t[0] = CompareNode(t[2], t[1], t[3])
        # print('compare generated: ', t[1], t[2], t[3])
    except TypeError:
        result = 'SEMANTIC ERROR'



def p_expression_biop(t):
    '''expression : expression PLUS term
                  | expression MINUS term
                  | expression POWER term
                  | expression TIMES term
                  | expression DIVIDE2 term
                  | expression DIVIDE term
                  | expression MODULUS term
                  | expression OR term
                  | expression AND term
                  | expression IN expression
                  '''
    try:
        # print('bip')
        t[0] = BopNode(t[2], t[1], t[3])
        # print('\n\n\n\n$$$ 3. bopnode assigned? ', t[0].evaluate(), t[1], t[2], t[3])
        # print('compare node. t[1]', t[1].evaluate())
    except TypeError:
        global result
        result = 'SEMANTIC ERROR'

def p_expression_biop_expr(t):
    '''expression : term PLUS expression
                  | term MINUS expression
                  | term POWER expression
                  | term TIMES expression
                  | term DIVIDE2 expression
                  | term DIVIDE expression
                  | term MODULUS expression
                  | term OR expression
                  | term AND expression
                  '''
    try:
        # print('bip')
        t[0] = BopNode(t[2], t[1], t[3])
        # print('\n\n\n\n$$$ 4. bopnode 2 assigned? ', t[0].evaluate(), t[1], t[2], t[3])
        # print('compare node. t[1]', t[1].evaluate())
    except TypeError:
        global result
        result = 'SEMANTIC ERROR'




# could be array or dictionary
def p_assign2_smt(t):
    '''
        expression : VARNAME SQOPEN primary SQCLOSE ASSIGN term SEMICOLON
    '''
    global variables
    global result
    try:
        # print('>> p_assign2_smt ')
        if(type(t[1]) == type(NumberNode('1'))):
            # print('left side 2 is NumberNode. ')
            t[1].value = t[6].value
            # result = 'SYNTAX ERROR'
            # t[0] = 'SYNTAX ERROR'
        else:
            if (t[1] in variables.keys()):
                t[0] = variables[t[1]]
                if(type(t[0]) == type('') or type(t[0]) == type(1) or type(t[0]) == type([])):
                    t[0] = t[0]
                else:
                    t[0] = t[0].evaluate()
                t[0][int(t[3].value)] = t[6]
            else:
                result = 'SEMANTIC ERROR'
    except TypeError:
        result = 'SEMANTIC ERROR'


def p_assign_smt(t):
    '''
        expression : VARNAME ASSIGN expression SEMICOLON
    '''
    global variables
    global result
    global currRunModeFunction
    global currRunMode
    try:
        # print('>> p_assign_smt ', t[1],t[2], t[3], t[4])

        if currRunMode == 0:

            if type(t[1]) == type(''):
                # print('>> var_assign.  new variable created : \n', t[1], t[3])
                # there is no variable, create new one

                if (type(t[3]) == type(BopNode(1, 1, 1))):
                    if (type(t[3].evaluate()) == type(1)):
                        variables[t[1]] = NumberNode(str(t[3].evaluate()))
                    elif (type(t[3].evaluate()) == type('')):
                        variables[t[1]] = StringNode(str(t[3].evaluate()))
                    elif (type(t[3].evaluate()) == type([])):
                        variables[t[1]] = ArrayNode(t[3].evaluate())
                    t[3] = t[3].evaluate()
                else:
                    variables[t[1]] = t[3]
                t[0] = t[3]
                # print('>> var_assign.  new variable created - result : \n', t[0], t[1], t[3])


            elif(type(t[1]) == type(NumberNode('1')) or type(t[1]) == type(StringNode('1'))):
                t[1].value = t[3].value
            else:
                if(type(t[3]) == type(BopNode(1,1,1))):
                    if(type(t[3].evaluate()) == type(1)):
                        variables[t[1]] = NumberNode(str(t[3].evaluate()))
                    elif(type(t[3].evaluate()) == type('')):
                        variables[t[1]] = StringNode(str(t[3].evaluate()))
                    elif (type(t[3].evaluate()) == type([])):
                        variables[t[1]] = ArrayNode(t[3].evaluate())
                    t[0] = t[3]
                    # print('\n\n\n\n$$$ 1. bopnode assigned? ', t[0], t[3])
        else:
            # function mode
            if type(t[1]) == type(''):
                # print('>> var_assign.  new variable created : \n', t[1], t[3])
                # there is no variable, create new one

                if (type(t[3]) == type(BopNode(1, 1, 1))):
                    if (type(t[3].evaluate()) == type(1)):
                        currRunModeFunction.localVariables[t[1]] = NumberNode(str(t[3].evaluate()))
                    elif (type(t[3].evaluate()) == type('')):
                        currRunModeFunction.localVariables[t[1]] = StringNode(str(t[3].evaluate()))
                    elif (type(t[3].evaluate()) == type([])):
                        currRunModeFunction.localVariables[t[1]] = ArrayNode(t[3].evaluate())
                    t[3] = t[3].evaluate()
                else:
                    currRunModeFunction.localVariables[t[1]] = t[3]
                t[0] = t[3]
                # print('>> function mode. var_assign.  new variable created - result : \n', t[0], t[1], t[3])


            elif(type(t[1]) == type(NumberNode('1')) or type(t[1]) == type(StringNode('1'))):
                t[1].value = t[3].value
            else:
                if(type(t[3]) == type(BopNode(1,1,1))):
                    if(type(t[3].evaluate()) == type(1)):
                        currRunModeFunction.localVariables[t[1]] = NumberNode(str(t[3].evaluate()))
                    elif(type(t[3].evaluate()) == type('')):
                        currRunModeFunction.localVariables[t[1]] = StringNode(str(t[3].evaluate()))
                    elif (type(t[3].evaluate()) == type([])):
                        currRunModeFunction.localVariables[t[1]] = ArrayNode(t[3].evaluate())
                    t[0] = t[3]


    except TypeError:
        result = 'SEMANTIC ERROR'


def p_term_assign_smt(t):
    '''
        expression : term ASSIGN expression SEMICOLON
    '''
    global variables
    global result
    try:
        # print('>> term_assign.  term : ', t[1])
        if type(t[1]) == type(''):
            # there is no variable, create new one
            if (type(t[3]) == type(BopNode(1, 1, 1))):
                if (type(t[3].evaluate()) == type(1)):
                    variables[t[1]] = NumberNode(str(t[3].evaluate()))
                elif (type(t[3].evaluate()) == type('')):
                    variables[t[1]] = StringNode(str(t[3].evaluate()))
                elif (type(t[3].evaluate()) == type([])):
                    variables[t[1]] = ArrayNode(t[3].evaluate())
            else:
                variables[t[1]] = t[3]
            t[0] = t[3]
        elif(type(t[1]) == type(NumberNode('1')) or type(t[1]) == type(StringNode('1'))):
            t[1].value = t[3].value
        else:
            if(type(t[3]) == type(BopNode(1,1,1))):
                if(type(t[3].evaluate()) == type(1)):
                    variables[t[1]] = NumberNode(str(t[3].evaluate()))
                elif(type(t[3].evaluate()) == type('')):
                    variables[t[1]] = StringNode(str(t[3].evaluate()))
                elif (type(t[3].evaluate()) == type([])):
                    variables[t[1]] = ArrayNode(t[3].evaluate())
                t[0] = t[3]
            # print('\n\n\n\n$$$ 2. bopnode assigned? ', t[0], t[3])
    except TypeError:
        result = 'SEMANTIC ERROR'







def p_expression_not(t):
    'expression : NOT term'
    try:
        if(t[2]):
            t[0] = False
        else:
            t[0] = True
    except TypeError:
        global result
        result = 'SEMANTIC ERROR'

def p_expression_term(t):
    'expression : term'
    try:
        t[0] = t[1]
    except TypeError:
        global result
        result = 'SEMANTIC ERROR'

def p_term_compare(t):
    '''term : term LESSE term
              | term GREATE term
              | term UNEQUAL term
              | term LESS term
              | term GREAT term
              | term EQUAL term
              '''
    try:
        t[0] = CompareNode(t[2], t[1], t[3])

    except TypeError:
        global result
        result = 'SEMANTIC ERROR'


def p_exprList_priority(t):
    'term : LPAREN expression RPAREN'
    t[0] = t[2]
    # print('exprList () ')

# returns array or dictionary's element
def p_term_specific_singleList2(t):
    'term : VARNAME SQOPEN expression SQCLOSE'
    global variables

    if (t[1] in variables.keys()):
        t[0] = variables[t[1]][t[3].evaluate()]
    else:
        global result
        result = 'SEMANTIC ERROR'
        # print('there is no array')

def p_term_specific_2dList(t):
    'term : VARNAME SQOPEN primary SQCLOSE SQOPEN primary SQCLOSE'
    global variables
    if (t[1] in variables.keys()):
        t[0] = variables[t[1]][int(t[3].value)][int(t[6].value)]
    else:
        global result
        result = 'SEMANTIC ERROR'
        # print('there is no array')

def p_term_specific_singleList(t):
    'term : VARNAME SQOPEN primary SQCLOSE'
    global variables
    if (t[1] in variables.keys()):
        t[0] = variables[t[1]][int(t[3].value)]
    else:
        global result
        result = 'SEMANTIC ERROR'


def p_term_singleList(t):
    'term : SQOPEN primary SQCLOSE'
    try:
        t[0] = [t[2]]
    except TypeError:
        global result
        result = 'SEMANTIC ERROR'


def p_term_list_index(t):
    'term : SQOPEN factor SQCLOSE SQOPEN NUMBER SQCLOSE'
    try:
        t[0] = t[2][t[5].value]
        # print('>>>> access array', t[0].value)
    except TypeError:
        global result
        result = 'SEMANTIC ERROR'

def p_term_list(t):
    'term : SQOPEN factor SQCLOSE'
    try:
        t[0] = t[2]
    except TypeError:
        global result
        result = 'SEMANTIC ERROR'


def p_term_emptylist(t):
    'term : SQOPEN SQCLOSE'
    try:
        t[0] = []
    except TypeError:
        global result
        result = 'SEMANTIC ERROR'


def p_term_primary_term(t):
    'factor : terms COMMA factor'
    try:
        if(type(t[3]) != type([]) and type(t[1]) == type(t[3])):
            t[0] = [t[1], t[3]]
        elif(type(t[3]) == type([]) and type(t[1]) != type(t[3])):
            t[0] = [t[1]] + t[3]
        elif(type(t[1]) == type([]) and type(t[1]) != type(t[3])):
            t[0] = [t[1], t[3]]
        elif(type(t[1]) == type([]) and type(t[1]) == type(t[3])):
            if(t[3]==[] and t[1]==[]):
                t[0] = [[], []]
            elif(t[1]==[]):
                t[0] = []
                t[0].append([])
                t[0] += t[3]
            elif(t[3]==[]):
                t[0] = t[1]
                t[0].append([])
            else:
                t[0] = t[1] + t[3]
        else:
            t[0] = [t[1], t[3]]
    except TypeError:
        global result
        result = 'SEMANTIC ERROR'

def p_terms_list(t):
    'terms : SQOPEN factor SQCLOSE'
    t[0] = [t[2]]

def p_terms_emptylist(t):
    'terms : SQOPEN SQCLOSE'
    try:
        t[0] = [[]]
    except TypeError:
        global result
        result = 'SEMANTIC ERROR'


def p_terms_not(t):
    'terms : NOT terms'
    try:
        if(t[2]):
            t[0] = False
        else:
            t[0] = True
    except TypeError:
        global result
        result = 'SEMANTIC ERROR'

def p_terms_biop(t):
    '''terms : terms PLUS terms
                  | terms MINUS terms
                  | terms POWER terms
                  | terms TIMES terms
                  | terms DIVIDE2 terms
                  | terms DIVIDE terms
                  | terms MODULUS terms
                  | terms OR terms
                  | terms AND terms
                  | terms IN terms
                  '''
    global result
    try:
        if t[2] == '+': t[0] = [t[1][0] + t[3][0]]
        elif t[2] == '-': t[0] = [t[1][0] - t[3][0]]
        elif t[2] == '**': t[0] = [t[1][0] ** t[3][0]]
        elif t[2] == '*': t[0] = [t[1][0] * t[3][0]]
        elif t[2] == '//': t[0] = [t[1][0] // t[3][0]]
        elif t[2] == '/': t[0] = [t[1][0] / t[3][0]]
        elif t[2] == '%': t[0] = [t[1][0] % t[3][0]]
        elif t[2] == 'or': t[0] = [t[1][0] or t[3][0]]
        elif t[2] == 'and': t[0] = [t[1][0] and t[3][0]]
        elif t[2] == 'in': t[0] = [t[1][0] in t[3][0]]
    except TypeError:
        # result = 'SEMANTIC ERROR'
        t[0] = BopNode(t[2], t[1][0], t[3][0]).evaluate()
        if type(t[0]) == type(0):
            t[0] = NumberNode(str(t[0]))
        elif type(t[0]) == type(''):
            t[0] = StringNode(str(t[0]))

def p_terms_compare(t):
    '''terms : terms LESSE terms
              | terms GREATE terms
              | terms UNEQUAL terms
              | terms LESS terms
              | terms GREAT terms
              | terms EQUAL terms
              '''
    try:
        if t[2] == '<=':
            t[0] = [t[1] <= t[3]]
        elif t[2] == '>=':
            t[0] = [t[1] >= t[3]]
        elif t[2] == '<>':
            t[0] = [t[1] != t[3]]
        elif t[2] == '<':
            t[0] = [t[1] < t[3]]
        elif t[2] == '>':
            t[0] = [t[1] > t[3]]
        elif t[2] == '==':
            t[0] = [t[1] == t[3]]
    except TypeError:
        t[0] = CompareNode(t[2], t[1][0], t[3][0]).evaluate()

def p_terms_priority(t):
    'terms : LPAREN terms RPAREN'
    try:
        t[0] = t[2]
        print('term () ')
    except TypeError:
        global result
        result = 'SEMANTIC ERROR'


def p_terms_primary(t):
    'terms : primary'
    try:
        t[0] = [t[1]]
    except TypeError:
        global result
        result = 'SEMANTIC ERROR'


def p_factor_terms(t):
    'factor : terms'
    try:
        t[0] = t[1]
    except TypeError:
        global result
        result = 'SEMANTIC ERROR'



def p_factor_primary(t):
    'factor : primary'
    try:
        t[0] = t[1]
        print('factor : primary', t[0])
    except TypeError:
        global result
        result = 'SEMANTIC ERROR'


def p_term_primary(t):
    'term : primary'
    try:
        t[0] = t[1]
    except TypeError:
        global result
        result = 'SEMANTIC ERROR'


def p_primary_string(t):
    'primary : STRING'
    try:
        print('primary: string ', t[1])
        t[0] = t[1]
    except TypeError:
        global result
        result = 'SEMANTIC ERROR'


def p_primary_varname(t):
    'primary : VARNAME'
    global variables
    global result
    global currRunModeFunction

    try:
        if currRunMode == 0:
            if(t[1] in variables.keys()):
                t[0] = variables[t[1]]
                # print('primary - varname found: ', t[0], t[1])
            else:
                t[0] = t[1]
                # print('primary - varname cant found: ', t[0], t[1])
                # print('current variables dict keys: ', variables.keys())
        else:
            if (t[1] in currRunModeFunction.localVariables.keys()):
                t[0] = currRunModeFunction.localVariables[t[1]]
                # print('primary - varname localVariables found: ', t[1], '=', t[0])
            else:
                t[0] = t[1]
                # print('primary - varname  localVariables cant found: ', t[0], t[1])
                # print('current variables dict keys: ', currRunModeFunction.localVariables.keys())
    except TypeError:

        result = 'SEMANTIC ERROR'


def p_primary_number(t):
    'primary : NUMBER'
    t[0] = t[1]


def p_primary_real(t):
    'primary : REAL'
    t[0] = t[1]


def p_error(t):
    global result
    result = 'SYNTAX ERROR'

import ply.yacc as yacc
parser = yacc.yacc()


inputFile  = open(sys.argv[1], 'r')

currBlockNumber = 0
blockDepth = 0
isClosed = False
singleLine = False




isValidProgram = True
curlyOpenCount = 0
curlyCloseCount = 0

currRunMode = 0
currRunModeFunction = None

# Check Valid Curly Braces
for line in inputFile:
    if line != None and line != '\n':
        curlyOpenCount += line.count('{')
        curlyCloseCount += line.count('}')

if curlyCloseCount != curlyOpenCount:
    isValidProgram = False
else:
    isValidProgram = True



inputFile  = open(sys.argv[1], 'r')

# Create Blocks
currBlock = None
# 0 is normal, 1 is function
currBlockType = 0

for line in inputFile:
    if line != None and line != '\n':
        # if there is '{  }', make new block, save statement, stay current block

        hasOpenCurlyBrace = '{' in line
        hasCloseCurlyBrace = '}' in line

        line = line.replace('\t', '')
        line = line.replace('\n', '')

        if hasOpenCurlyBrace and hasCloseCurlyBrace:

            if 'else' in line:
                if currBlock == None:
                    isValidProgram = False
                    break
                newBlock = BlockNode(currBlock.parent)
                currBlock.parent.addStm(newBlock)
                currBlock = newBlock
                currBlock.addStm(line)

            elif '(' in line and ')' in line:
                # Check block-type
                if 'if' in line or 'while' in line:
                    tempLine = line.replace(' ', '')
                    tempLine = tempLine.replace('\t', '')
                    tempLine = tempLine.split('(')
                    
                    if tempLine[0] == 'if' or tempLine[0] == 'while':
                        if currBlock == None:
                            blocks.append(BlockNode(None))
                            currBlock = blocks[-1]
                        else:
                            newBlock = BlockNode(currBlock)
                            currBlock.addStm(newBlock)
                    else:
                        newBlock = FunctionNode(currBlock)
                        functionBlocks[str(tempLine[0])] = newBlock

                        # function arguments
                        argLine = tempLine[1]
                        argLine = argLine.replace(')', '')
                        argLine = argLine.replace('{', '')
                        argLine = argLine.replace('\t', '')
                        argLine = argLine.replace('\n', '')
                        argLine = argLine.split(',')
                        for i in range(len(argLine)):
                            newBlock.setArgument(argLine[i], None)


                        # currBlockType = 1   # closed block immediately
                    currBlock = newBlock
                    currBlock.addStm(line)
                else:
                    if currBlock == None:
                        blocks.append(BlockNode(None))
                        currBlock = blocks[-1]
                    else:
                        newBlock = BlockNode(currBlock)
                        currBlock.addStm(newBlock)
                        currBlock = newBlock
                    currBlock.addStm(line)
            else:
                if currBlock == None:
                    blocks.append(BlockNode(None))
                    currBlock = blocks[-1]
                else:
                    newBlock = BlockNode(currBlock)
                    currBlock.addStm(newBlock)
                    currBlock = newBlock
                currBlock.addStm(line)

            # function ends on singleline
            # multiple blocks on singleline

        elif hasOpenCurlyBrace and not hasCloseCurlyBrace:

            if '(' in line and ')' in line:
                # Check block-type
                tempLine = line.replace(' ', '')
                tempLine = tempLine.replace('\t', '')
                tempLine = tempLine.split('(')

                if 'if' in line or 'while' in line:
                    if tempLine[0] == 'if' or tempLine[0] == 'while':
                        if currBlock == None:
                            blocks.append(BlockNode(None))
                            currBlock = blocks[-1]
                        newBlock = BlockNode(currBlock)
                        currBlock.addStm(newBlock)
                    else:
                        newBlock = FunctionNode(currBlock)
                        functionBlocks[str(tempLine[0])] = newBlock
                        currBlockType = 1
                else:
                    newBlock = FunctionNode(currBlock)
                    functionBlocks[str(tempLine[0])] = newBlock
                    currBlockType = 1

                    # argument check
                    argLine = tempLine[1]
                    argLine = argLine.replace(')', '')
                    argLine = argLine.replace('{', '')
                    argLine = argLine.replace('\t', '')
                    argLine = argLine.replace('\n', '')

                    argLine = argLine.split(',')
                    for i in range(len(argLine)):
                        newBlock.setArgument(argLine[i], None)


            else:
                if 'else' in line:
                    if currBlock == None:
                        isValidProgram = False
                        break
                if currBlock == None:
                    blocks.append(BlockNode(None))
                    newBlock = blocks[-1]
                else:
                    newBlock = BlockNode(currBlock)
                    currBlock.addStm(newBlock)
            currBlock = newBlock
            currBlock.addStm(line)

        elif hasCloseCurlyBrace and not hasOpenCurlyBrace:
            if currBlock == None:
                isValidProgram = False
                break

            # this closing curlybrace is closing function block
            if(currBlockType == 1 and currBlock.parent == None):
                currBlockType = 0
                if len(blocks) > 0:
                    currBlock = blocks[-1]
                else:
                    currBlock = None
            else:
                currBlock = currBlock.parent
        else:
            if currBlock != None:
                currBlock.addStm(line)


# print('Number of blocks: ', len(blocks))
# print('Number of functionBlocks: ', len(functionBlocks))


def recurseBlocks(currB):
    global result

    skipNumber = 9999

    if currB == None:
        return

    if result != None:
        # print('2 recurseBlock result: ', result)
        return

    if type(currB) == type(BlockNode(None)):

        isIf = False
        isElse = False
        isWhile = False
        targetNode = None

        for i in range(len(currB.childNodes)):

            # print('\n [debug] >> under block currNode:', i, '\n', currB.childNodes[i])
            currNode = currB.childNodes[i]

            # if type(currNode) != type(''):
                # print('[[[ currNode]]] : ', currNode.childNodes)

            if result != None:
                # print('1 recurseBlock result: ', result)
                break

            if i == skipNumber:
                # print('skipNumber')
                continue

            if type(currNode) != type(''):
                # if it is not blocktype, but it could be single statement if, else, while statement

                # print('[[[ currNode]]] : ', currNode)

                if type(currNode.childNodes[0]) == type(''):
                    currNode = currNode.childNodes[0]

                    if 'if' in currNode:
                        isIf = True

                        # if/else block
                        if i + 1 < len(currB.childNodes):
                            if type(currB.childNodes[i + 1]) == type(BlockNode(None)):
                                currline = currB.childNodes[i + 1].childNodes[0]
                                if (type(currline) == type('')) and ('else' in currline):
                                    isElse = True

                    elif 'else' in currNode:    # single else block
                        isElse = True
                        print('else')

                    elif 'while' in currNode:
                        isWhile = True


                    if isElse == True and isIf == True: # if/else blocks. Choose a block to run

                        # print('if/else')

                        ifStatement = currB.childNodes[i].childNodes[0]

                        if ('}' in ifStatement):
                            ifStatement = ifStatement.replace('}', '')
                        if ('{' in ifStatement):
                            ifStatement = ifStatement.replace('{', '')
                        ifStatement_result = parser.parse(ifStatement)

                        skipNumber = i + 1 # because I run one of block now
                        if(ifStatement_result == True):
                            targetNode = currB.childNodes[i]
                        else:
                            targetNode = currB.childNodes[i+1] # run else block

                    elif isElse == False and isIf == True: # run if block no matter what

                        ifStatement = currB.childNodes[i].childNodes[0]

                        if ('}' in ifStatement):
                            ifStatement = ifStatement.replace('}', '')
                        if ('{' in ifStatement):
                            ifStatement = ifStatement.replace('{', '')

                        ifStatement_result = parser.parse(ifStatement)
                        if (ifStatement_result == True):
                            targetNode = currB.childNodes[i]
                        else:
                            continue

                    elif isWhile == True: # run while loop until statement is False

                        targetNode = currB.childNodes[i]
                        whileStatement = currB.childNodes[i].childNodes[0]

                        if ('}' in whileStatement):
                            whileStatement = whileStatement.replace('}', '')
                        if ('{' in whileStatement):
                            whileStatement = whileStatement.replace('{', '')
                        # print('while :', whileStatement)
                        whileStatement_result = parser.parse(whileStatement)

                        while (whileStatement_result == True):
                            recurseBlocks(targetNode)
                            whileStatement_result = parser.parse(currB.childNodes[i].childNodes[0])
                        else:
                            continue
                    else:
                        targetNode = currB.childNodes[i]
                else:
                    targetNode = currNode

                recurseBlocks(targetNode)

            else:
                recurseBlocks(currNode)
    else:
        # statements
        # print('currB statement: ', currB)
        if len(currB) < 3:
            return
        ast = parser.parse(currB)
        # print('\n[debug] non-block stmt:', currB, 'non-block result: ', ast)


# print('curlyOpenCount', curlyOpenCount)
# print('curlyCloseCount', curlyCloseCount)


if isValidProgram == True:
    for currentBlock in blocks:

        if result != None:
            print('SEMANTIC ERROR')
            break

        if type(currentBlock) != type(BlockNode(None)) and type(currentBlock) != type(FunctionNode(None)):
            print('SYNTAX ERROR')
            print(currentBlock)
            break

        # for node in currentBlock.childNodes:
        if result == None:
            recurseBlocks(currentBlock)
        else:
            print('SEMANTIC ERROR')
            break

else:
    print('SYNTAX ERROR')

if result != None:
    print(result)

inputFile.close()