"""
Fusion Core (*.fs) [Version 0.1.0-alpha] - Dev Phase

Description:
Fusion is a programming language that aims to blends Python's simplicity,
C++'s speed, Rust's safety, and Go's concurrency to deliver high-performance and clear code.
Developed by Aditya Gaur (xditya: https://github.com/xdityagr)

"""

# -> Check grammar rules for understanding the code better.

import string, sys, os
from utils import *

# CONSTANTS 

# Digits, Letters
DIGITS = '0123456789'
LETTERS = string.ascii_letters
LETTERS_DIGITS = LETTERS + DIGITS

# Language Keywords
KEYWORDS =  [
                'let', 
                'and', 
                'or', 
                'not',  
                'if',
                'elif',
                'else', 
                'for',
                'while',
                'in',
                '..',
                'fun'
            ]

# Language variable types, probably shouldn't be stored in a list.
VARTYPES = ['Dict', 'Bool', 'Int', 'Float', 'String', 'List', 'Dict', 'Set']

# Token Types
TT_INT = 'INT'
TT_STR = 'STR'

TT_PLUS = 'PLUS'
TT_MINUS = 'MINUS'
TT_DIV = 'DIV'
TT_MUL = 'MUL'
TT_FLOAT = 'FLOAT'

TT_LPAREN = 'LPAREN' # (
TT_RPAREN = 'RPAREN' # )

TT_LCPAREN = 'LCPAREN' # {
TT_RCPAREN = 'RCPAREN' # } 

TT_LSPAREN = 'LSPAREN' # [
TT_RSPAREN = 'RSPAREN' # ]

TT_LSHIFT = 'LSHIFT' # >> opr; Use: List << Value:Int, List, String, Bool; Eequivalent to append() in python
TT_RSHIFT = 'RSHIFT' # >> opr; Use: List >> Value:Int; Eequivalent to pop() in python

TT_POW = 'POW' # **
TT_EOF = 'EOF' # End of file

TT_VARIDENTIFIER = 'IDENTIFIER' # Variable Identifier; let <identifier> 

TT_KWORD = 'KEYWORD' # Defined in keywords list
TT_EQ = 'EQ' # = 
TT_VARTYPEASSIGN = 'TYPEASSIGN' # ':' to assign/annotate a type; Use: let a : Int = 10
TT_VARTYPE = 'VARTYPE' # Variable types defined in list

# Comparisons
TT_EE = 'EE'  # Equivalence (==)
TT_NE = 'NE'  # Not equals (!=)
TT_LT  = 'LT' # Less than  (<)
TT_GT = 'GT'  # Greater than (>)
TT_LTE = 'LTE'# Less than equals (<=)
TT_GTE = 'GTE'# Greater than equals (>=)

TT_RANGE = 'RANGE' # '..' ; Use: x..y;  Equivalent to range(x,y) in python.
TT_TILDA = 'TILDA' # ~ (For anonymous functions)
TT_COMMA = 'COMMA' 

TT_NEWLINE = 'NEWLINE' 

ESC_CHARS = {   'n':'\n', 
                't':'\t'
            }

# ERROR CLASSES
class Error:
    def __init__(self, pos_start, pos_end, error_name, details):
        
        self.pos_start:Traceback = pos_start
        self.pos_end:Traceback = pos_end

        self.error_name = error_name
        self.details = details

    def as_string(self):
        # print(f"{self.pos_start.idx=}, {self.pos_end.idx=}")
        error_details = f"Traceback: File '{self.pos_start.fn}', line {self.pos_start.lno + 1}\n{self.indicate_error_char(self.pos_start.fctx, self.pos_start, self.pos_end)}\n{self.error_name} : {self.details}\n"
        return error_details
        
    def indicate_error_char(self, text, pos_start, pos_end):
        result = ' '

        # Calculate indices
        idx_start = max(text.rfind('\n', 0, pos_start.idx), 0)
        idx_end = text.find('\n', idx_start + 1)
        if idx_end < 0: idx_end = len(text)
        
        # Generate each line
        line_count = pos_end.lno - pos_start.lno + 1
        for i in range(line_count):
            # Calculate line columns
            line = text[idx_start:idx_end]
            col_start = pos_start.colno if i == 0 else 0
            col_end = pos_end.colno if i == line_count - 1 else len(line) - 1

            # Append to result
            result += line + '\n'
            result += ' ' * col_start + '^' * (col_end - col_start)

            # Re-calculate indices
            idx_start = idx_end
            idx_end = text.find('\n', idx_start + 1)
            if idx_end < 0: idx_end = len(text)

        return result.replace('\t', '')
        
   
class IllegalCharacterError(Error):
    def __init__(self, pos_start, pos_end, details):
        super().__init__(pos_start, pos_end,'Illegal Character Error', details)

class InvalidSyntaxError(Error):
    def __init__(self, pos_start, pos_end, details):
        super().__init__(pos_start, pos_end,'Invalid Syntax Error', details)

class RuntimeError(Error):
    def __init__(self, pos_start, pos_end, details, context):
        super().__init__(pos_start, pos_end,'Runtime Error', details)
        self.context = context

    def as_string(self):
        error_details = f"Traceback: {'\n' + self.generate_traceback()}\nLocation:\n{self.indicate_error_char(self.pos_start.fctx, self.pos_start, self.pos_end)}\n{self.error_name} : {self.details}\n"
        return error_details

    def generate_traceback(self):
        trace = ''
        pos = self.pos_start
        ctx = self.context

        while ctx:
            trace = f' File {pos.fn}, line {str(pos.lno + 1)}, in {ctx.display_name}\n'  + trace
            pos = ctx.parent_entry_pos
            ctx = ctx.parent 

        return trace
    
class ExpectedCharacterError(Error):
    def __init__(self, pos_start, pos_end, details):
        super().__init__(pos_start, pos_end,'Expected Character Error', details)
        
# TRACEBACK CLASS
class Traceback:
    def __init__(self, filename, file_ctx, index, line_no, col_no):
        self.fn= filename
        self.fctx = file_ctx
        self.idx = index
        self.lno = line_no 
        self.colno = col_no

    def advance(self, current_char=None):
        self.idx += 1
        self.colno += 1

        if current_char == '\n':
            self.lno += 1
            self.colno = 0

        return self
    
    def copy(self): 
        return Traceback(self.fn, self.fctx, self.idx, self.lno, self.colno)


# LEXICAL ANALYSIS  : Tokenize the text, break it down into tokens to further make sense out of it when parsing.
class Token:
    def __init__(self, type_, value=None, pos_start=None, pos_end = None):
        self.type = type_
        self.value = value 
        if pos_start: 
            self.pos_start = pos_start.copy()
            self.pos_end = pos_start.copy()
            self.pos_end.advance()

        if pos_end: self.pos_end = pos_end

    def matches(self, type_, value):
        return self.type == type_ and self.value == value

    def __repr__(self):
        if self.value: return f"{self.type}:{self.value}"
        return f"{self.type}"
    
# LEXER
class Lexer:
    def __init__(self, filename, file_txt):
        self.fn = filename
        self.text = file_txt
        self.pos = Traceback(self.fn, self.text, -1, 0, -1)

        self.curr_char = None
        self.advance()

    def advance(self):
        self.pos.advance(self.curr_char)
        self.curr_char = self.text[self.pos.idx] if self.pos.idx < len(self.text) else None
    
    def tokenize(self):
        tokens = []
    
        while self.curr_char != None:
            if self.curr_char in ' \t':
                self.advance()
            
            elif self.curr_char in ';\n':
                tokens.append(Token(TT_NEWLINE, pos_start=self.pos))
                self.advance()

            elif self.curr_char in DIGITS:
                tokens.append(self.tokenize_digits())
            
            elif self.curr_char in LETTERS:
                tokens.append(self.tokenize_identifer())

            elif self.curr_char == '.':
                token, error = self.tokenize_range()
                if error: return [], error 
                tokens.append(token)

            elif self.curr_char in ("'", '"'):
                token, error = self.tokenize_string()

                if error: return [], error 
                tokens.append(token)
        
            elif self.curr_char == '/':
                self.create_comment()
            elif self.curr_char == '+':
                tokens.append(Token(TT_PLUS, pos_start=self.pos))
                self.advance()

            elif self.curr_char == '-':
                tokens.append(Token(TT_MINUS, pos_start=self.pos))
                self.advance()

            elif self.curr_char == '/':
                tokens.append(Token(TT_DIV, pos_start=self.pos))
                self.advance()

            elif self.curr_char == '(':
                tokens.append(Token(TT_LPAREN, pos_start=self.pos))
                self.advance()
            elif self.curr_char == ')':
                tokens.append(Token(TT_RPAREN, pos_start=self.pos))
                self.advance()

            elif self.curr_char == '{':
                tokens.append(Token(TT_LCPAREN, pos_start=self.pos))
                self.advance()

            elif self.curr_char == '}':
                tokens.append(Token(TT_RCPAREN, pos_start=self.pos))
                self.advance()

            elif self.curr_char == '[':
                tokens.append(Token(TT_LSPAREN, pos_start=self.pos))
                self.advance()

            elif self.curr_char == ']':
                tokens.append(Token(TT_RSPAREN, pos_start=self.pos))
                self.advance()

            elif self.curr_char == ':':
                tokens.append(Token(TT_VARTYPEASSIGN, pos_start=self.pos))
                self.advance()

            elif self.curr_char == '*':
                tokens.append(self.tokenize_power())

            elif self.curr_char == '!':
                token, error = self.tokenize_notequals()
                if error: return [], error 
                tokens.append(token)

            elif self.curr_char == '=':
                tokens.append(self.tokenize_equals())

            elif self.curr_char == '<':
                tokens.append(self.tokenize_lessthan_or_lshift())

            elif self.curr_char == '>':
                tokens.append(self.tokenize_greaterthan_or_rshift())

            elif self.curr_char == ',':
                tokens.append(Token(TT_COMMA, pos_start=self.pos))
                self.advance()
            elif self.curr_char == '~':
                tokens.append(Token(TT_TILDA, pos_start=self.pos))
                self.advance()
            
            else:
                pos_start = self.pos.copy()
                char = self.curr_char
                self.advance()

                ill_char_error_str = f""" Encountered an Illegal Character, '{char}' in '{self.text}'"""
                return [], IllegalCharacterError(pos_start, self.pos, f"{ill_char_error_str}")
                
        tokens.append(Token(TT_EOF, pos_start=self.pos))
        return tokens, None
    
    def create_comment(self):
        self.advance()

        if self.curr_char == '/': # Single-line comment (//)
            while self.curr_char is not None and self.curr_char != '\n':
                self.advance()

        elif self.curr_char == '*': # Multi-line comment (/* stuff */)
            while self.curr_char is not None and self.curr_char != '*' :
                self.advance()
                self.advance()
                if self.curr_char == '/':
                    break

            self.advance()


    def tokenize_equals(self): 
        tt = TT_EQ
        pos_start = self.pos.copy()
        self.advance()

        if self.curr_char == '=':
            self.advance()
            tt = TT_EE
            
        return Token(tt, pos_start=pos_start, pos_end=self.pos)
        

    def tokenize_notequals(self): 
        pos_start = self.pos.copy()
        self.advance()

        if self.curr_char == '=':
            self.advance()
            return Token(TT_NE, pos_start=pos_start, pos_end=self.pos), None

        self.advance()
        return None, ExpectedCharacterError(pos_start, self.pos, "Expected '=' after '!' ")

    def tokenize_greaterthan_or_rshift(self):         
        tt = TT_GT
        pos_start = self.pos.copy()
        self.advance()

        if self.curr_char == '=':
            self.advance()
            tt = TT_GTE
        elif self.curr_char == '>':
            self.advance()
            tt = TT_RSHIFT

        return Token(tt, pos_start=pos_start, pos_end=self.pos)
    
    def tokenize_lessthan_or_lshift(self): 
        tt = TT_LT
        pos_start = self.pos.copy()
        self.advance()

        if self.curr_char == '=':
            self.advance()
            tt = TT_LTE
        elif self.curr_char == '<':
            self.advance()
            tt = TT_LSHIFT

        return Token(tt, pos_start=pos_start, pos_end=self.pos)
        
    def tokenize_power(self):         
        tt = TT_MUL
        pos_start = self.pos.copy()
        self.advance()

        if self.curr_char == '*':
            self.advance()
            tt = TT_POW
            
        return Token(tt, pos_start=pos_start, pos_end=self.pos)
        

    
    def tokenize_range(self): 
        tt = TT_RANGE
        pos_start = self.pos.copy()
        self.advance()

        if self.curr_char == '.':
            self.advance()
            return Token(tt, pos_start=pos_start, pos_end=self.pos), None
        
        return None, ExpectedCharacterError(pos_start, self.pos, "Expected '.' after '.' to define a range ")

    def tokenize_string(self):
        string = ''
        
        start_col = self.curr_char

        pos_start = self.pos.copy()
        escape_char = False
        self.advance()
        
        while self.curr_char != None and (self.curr_char != start_col or escape_char):
            if escape_char:
                string += ESC_CHARS.get(self.curr_char, self.curr_char)
                self.advance()  
                escape_char = False  
            else:
                if self.curr_char == '\\':
                    escape_char = True 
                else:
                    string += self.curr_char
                self.advance()
                

        if self.curr_char != start_col:
            return None, ExpectedCharacterError(pos_start, self.pos, f"Expected {start_col} ")
        
        self.advance()
        return Token(TT_STR, string, pos_start=pos_start, pos_end=self.pos), None
    
    def tokenize_identifer(self):
        id_str = ''
        pos_start = self.pos.copy()

        while self.curr_char != None and self.curr_char in LETTERS_DIGITS + '_':
            id_str += self.curr_char
            self.advance()

        if id_str in KEYWORDS : 
            tok_type = TT_KWORD

        elif id_str in  VARTYPES : tok_type = TT_VARTYPE
        else:
            tok_type = TT_VARIDENTIFIER
    
        return Token(tok_type, id_str, pos_start, self.pos)
        

    def tokenize_digits(self):
        num_str = ''
        dot_count = 0   
        pos_start = self.pos.copy() 

        while self.curr_char != None and self.curr_char in DIGITS + '.':
            if self.curr_char == '.':   
                if dot_count == 1: break
                
                next_pos = self.pos.copy()
                next_pos.advance(self.curr_char)
                next_char = self.text[next_pos.idx] if next_pos.idx < len(self.text) else None
                if next_char == '.':
                    break
                dot_count += 1
                num_str += self.curr_char
            else:
                num_str += self.curr_char
            self.advance()

        if dot_count == 0:
            return Token(TT_INT, int(num_str), pos_start=pos_start, pos_end=self.pos)

        else:
            return Token(TT_FLOAT, float(num_str), pos_start=pos_start, pos_end=self.pos)
        

# NODES : Keeps track of different types, their values and their exact position in text for AST and AST repr gen
class NumberNode:
    def __init__(self, token):
        self.token = token # The value stored
        self.pos_start=token.pos_start # char start pos
        self.pos_end= token.pos_end # char end pos

    def __repr__(self):
        return f'{self.token}'

class StringNode:
    def __init__(self, token):
        self.token = token
        
        self.pos_start=token.pos_start
        
        self.pos_end= token.pos_end

    def __repr__(self):
        return f'{self.token}'
    

class ListNode:
    def __init__(self, element_nodes, pos_start, pos_end):
        self.elements = element_nodes  # The values being stored in a list, which is a list xd
        self.pos_start=pos_start # tried my best to not use core python built ins
        self.pos_end= pos_end


class IndexNode:
    def __init__(self, list_node, index_node, end=None, step=None):
        self.list_node = list_node  # The value being indexed
        self.index_node = index_node  # The single index/start of range
        self.end = end  # The end of the range; none if omitted
        self.step = step  # The step of the range; none if present

        self.pos_start = list_node.pos_start

        self.pos_end = step.pos_end if step else (end.pos_end if end else index_node.pos_end)

    @property
    def is_range(self):
        return self.end is not None

class IFNode:
    def __init__(self, cases, else_case):

        self.cases = cases # cases in the if statement
        self.else_case = else_case # tuple of else case
        
        self.pos_start= self.cases[0][0].pos_start # first case (tuple), first element of tuple (expr)
        self.pos_end= (self.else_case or self.cases[-1])[0] .pos_end

    
class BinOpNode: # Binary operation node
    def __init__(self, left_node, opr_token, right_node):
        
        self.left_node = left_node # left operand
        self.opr_token = opr_token # operator
        self.right_node = right_node # right operand

        self.pos_start=self.left_node.pos_start
        self.pos_end= self.right_node.pos_end

    def __repr__(self):
        return f'({self.left_node}, {self.opr_token}, {self.right_node})'
    
class UnaryOpNode: # unary operators -,+,not
    def __init__(self, opr_token, node):
        self.opr_token = opr_token
        self.node = node
        self.pos_start=self.opr_token.pos_start
        self.pos_end= self.node.pos_end

    def __repr__(self):
        return f'({self.opr_token}, {self.node})'
    

class VarAccessNode: # for accessing a defined var
    def __init__(self, var_name):
        self.var_name = var_name
        self.pos_start=self.var_name.pos_start
        self.pos_end= self.var_name.pos_end

class VarAssignNode :# for defining a var or assigning to a pre-defined var
    def __init__(self, var_name, var_type=None, var_value=None):
        
        self.var_name = var_name # var name
        self.var_type_node = var_type  # token or None, no inference

        self.value_node = var_value
        self.pos_start = self.var_name.pos_start
        if self.value_node:
            self.pos_end = self.value_node.pos_end
        elif self.var_type_node:
            self.pos_end = self.var_type_node.pos_end
        else:
            self.pos_end = self.var_name.pos_end

    def __repr__(self):
        return f"VarAssignNode('{self.var_name}:{self.var_type_node} = {self.value_node}') at Position {self.pos_start.idx}, {self.pos_end.idx}"
        
class ForNode: # For loop node
    def __init__(self, var_name_token, start_value_node, end_value_node, step_value_node, body_node, retr_void):

        self.var_name_token = var_name_token

        self.start_value_node = start_value_node
        self.end_value_node = end_value_node

        self.step_value_node= step_value_node

        self.body_node= body_node

        self.retr_void = retr_void # should it return void?

        self.pos_start = self.var_name_token.pos_start
        self.pos_end = self.body_node.pos_end

    def __repr__(self):
        return f"ForNode({self.var_name_token}, {self.start_value_node}..{self.end_value_node}..{self.step_value_node}, {self.body_node})"

class WhileNode: # while loop node
    def __init__(self, condition_node, body_node, retr_void):

        self.condition_node = condition_node
        self.body_node= body_node
        self.retr_void = retr_void

        self.pos_start = self.condition_node.pos_start
        self.pos_end = self.body_node.pos_end
    
class FunDefNode:  # fun definition node
    def __init__(self, var_name_token, arg_name_tokens, body_node, retr_void) :
        
        self.var_name_token = var_name_token
        self.arg_name_tokens = arg_name_tokens
        self.body_node = body_node
        self.retr_void= retr_void

        if self.var_name_token:
            self.pos_start = self.var_name_token.pos_start
        elif len(self.arg_name_tokens)>0:
            self.pos_start = self.arg_name_tokens[0].pos_start
        else:
            self.pos_start = self.body_node.pos_start

        self.pos_end = self.body_node.pos_end
    

class CallNode: # calling a fun
    def __init__(self, node_to_call, arg_nodes) :
        self.node_to_call = node_to_call
        self.arg_nodes = arg_nodes

        self.pos_start = self.node_to_call.pos_start

        if len(self.arg_nodes) > 0 :
            self.pos_end = self.arg_nodes[-1].pos_end
        else:
            self.pos_end = self.node_to_call.pos_end
        

# PARSE RESULT : Helps parser to hold parsed result, error, and handle advancements
class ParseResult:
    def __init__(self):
        self.error = None
        self.node = None
        self.last_registered_advance_count = 0

        self.advance_count = 0
        self.rollback_count = 0    

    def  success(self, node):
        self.node = node 
        return self
    
    def register_advancement(self): 
        self.last_registered_advance_count = 1
        self.advance_count += 1
    
    def register(self, res):
        self.last_registered_advance_count = res.advance_count

        self.advance_count += res.advance_count
        if res.error: self.error = res.error
        return res.node
        
    def try_register(self, res):
        if res.error:
            self.rollback_count = res.advance_count
            return None
        return self.register(res)
    
    def failure(self, error):
        if not self.error or self.last_registered_advance_count == 0:
            self.error = error
        return self

# PARSER: Parse the token stream and enforce the syntax rules defined in FusionGrammar.txt.
class Parser:
    def __init__(self, tokens, future=None):
        self.tokens = tokens 
        self.future = future
        self.token_idx = -1
        self.advance()

    def advance(self): # Advance to next token
        self.token_idx += 1 
        self.update_current_token()

        return self.current_token
    
    def rollback(self, step=1): # Revert to prev token
        self.token_idx -= step
        self.update_current_token()
        return self.current_token
    
    def update_current_token(self): # update token
        if self.token_idx < len(self.tokens):
            self.current_token = self.tokens[self.token_idx]
            
    def parse(self): # top level parsing method for a token stream
        res = self.statements()

        if not res.error and self.current_token.type != TT_EOF:
            return res.failure(InvalidSyntaxError(
                    self.current_token.pos_start, self.current_token.pos_end, 
                    "Expected '+', '-', '*', '/' or end of expression"
                ))

        return res

    def power(self): # ** 
        return self.bin_opr(self.call, (TT_POW, ), self.factor)
    
    def statements(self): # parse statements separated by NEWLINE or ;
        res = ParseResult()
        statements = []
        pos_start = self.current_token.pos_start.copy()

        while self.current_token.type == TT_NEWLINE: 
            res.register_advancement()
            self.advance()

        statement = res.register(self.expr())
        if res.error: return res
        statements.append(statement)
        
        more_statements = True

        while True:
            nline_count = 0
            while self.current_token.type == TT_NEWLINE: 
                res.register_advancement()
                self.advance()
                nline_count += 1

            if nline_count == 0: more_statements=False
            if not more_statements : break
            
            statement = res.try_register(self.expr())
            if not statement: 
                self.rollback(res.rollback_count)
                more_statements=False
                continue

            statements.append(statement)

        return res.success(ListNode(statements, pos_start, self.current_token.pos_end.copy()))

    def call(self): # parse function calls
        res = ParseResult()
        atom = res.register(self.atom())
        if res.error: return res 

        if self.current_token.type == TT_LPAREN: 
            res.register_advancement()
            self.advance()

            arg_nodes = []

            if self.current_token.type == TT_RPAREN: 
                res.register_advancement()
                self.advance()
            else:
                arg_nodes.append(res.register(self.expr()))
                if res.error: 
                    return res.failure(InvalidSyntaxError(
                    self.current_token.pos_start, self.current_token.pos_end, 
                    "Expected 'for', 'if','let', Int, Float, Identifier, '+', '-', '*', '/', '(' or ')'"
                ))
                
                while self.current_token.type == TT_COMMA:
                    res.register_advancement()
                    self.advance()

                    arg_nodes.append(res.register(self.expr()))
                    if res.error: return res

                if self.current_token.type != TT_RPAREN:
                    return res.failure(InvalidSyntaxError(
                        self.current_token.pos_start, self.current_token.pos_end,
                                            "Expected ',' or ')' "))     
                
                res.register_advancement()
                self.advance()

            return res.success(CallNode(atom, arg_nodes))
        return res.success(atom)


    def atom(self):  # Parse an atom (smallest syntactic unit of an expr)
        res = ParseResult()
        token = self.current_token

        if token.type in (TT_PLUS, TT_MINUS): 
            res.register_advancement()
            self.advance()
            atom = res.register(self.atom())
            if res.error: return res
            return res.success(UnaryOpNode(token, atom))
        
        elif token.type == TT_VARIDENTIFIER:
            res.register_advancement()
            self.advance()
            base_node = VarAccessNode(token)
        elif token.type in (TT_INT, TT_FLOAT):
            res.register_advancement()
            self.advance()
            base_node = NumberNode(token)
        elif token.type in TT_STR:
            res.register_advancement()
            self.advance()
            base_node = StringNode(token)
            
        elif token.matches(TT_KWORD, 'if'):
            if_expr = res.register(self.if_expr())
            if res.error: return res 
            return res.success(if_expr)
        elif token.matches(TT_KWORD, 'for'):
            for_expr = res.register(self.for_expr())
            if res.error: return res 
            return res.success(for_expr)
        elif token.matches(TT_KWORD, 'while'):
            while_expr = res.register(self.while_expr())
            if res.error: return res 
            return res.success(while_expr)
        elif token.matches(TT_KWORD, 'fun'):
            fun_def = res.register(self.fun_def())
            if res.error: return res 
            return res.success(fun_def)
        elif token.type == TT_TILDA:
            fun_def = res.register(self.anonymous_fun_def())
            if res.error: return res 
            return res.success(fun_def)
        
        elif token.type == TT_LSPAREN:
            list_expr = res.register(self.list_expr())
            if res.error: return res 
            base_node = list_expr

        elif token.type == TT_LPAREN:
            res.register_advancement()
            self.advance()
            expr = res.register(self.expr())
            if res.error: return res
            if self.current_token.type == TT_RPAREN:
                res.register_advancement()
                self.advance()
                base_node = expr
            else: 
                return res.failure(InvalidSyntaxError(self.current_token.pos_start, self.current_token.pos_end, "Expected ')'"))
        else:
            return res.failure(InvalidSyntaxError(token.pos_start, token.pos_end, "Expected 'for', 'if','let', Int, Float, Identifier, '+', '-', '*', '/', '[' or '('"))

        # parse a range, could've added it to another method-
        while self.current_token.type == TT_LSPAREN:
            res.register_advancement()
            self.advance()
            
            # Check for omitted start [..3]
            if self.current_token.type == TT_RANGE:
                start = None
                res.register_advancement()
                self.advance()
            else:
                # Parse start idx
                start = res.register(self.expr())
                if res.error: return res
                if self.current_token.type == TT_RANGE:
                    res.register_advancement()
                    self.advance()

            # Check for omitted end [1..]
            if self.current_token.type == TT_RSPAREN:
                end = None
            else:
                end = res.register(self.expr())
                if res.error: return res

            # Check for a step [1..5..2]
            step = None
            if self.current_token.type == TT_RANGE:
                res.register_advancement()
                self.advance()
                step = res.register(self.expr())
                if res.error: return res

            if self.current_token.type != TT_RSPAREN:
                return res.failure(InvalidSyntaxError(
                    self.current_token.pos_start, self.current_token.pos_end,
                    "Expected '..', ']'"
                ))
            res.register_advancement()
            self.advance()

            
            if start is not None or end is not None: # Slicing
                index_node = IndexNode(base_node, start, end=end, step=step)

            else: # Access
                index_node = IndexNode(base_node, start)

            base_node = index_node

        return res.success(base_node)

    def factor(self): # parse a factor composed of a power which is composed of a call which is composed of ultimately, an atom
        res = ParseResult()
        tok = self.current_token

        if tok.type in (TT_PLUS, TT_MINUS):
            res.register_advancement()
            self.advance()
            factor = res.register(self.factor())
            if res.error: return res
            return res.success(UnaryOpNode(tok, factor))

        return self.power()         
    
    def term(self):  # parse multiple factors together, making a term
        return self.bin_opr(self.factor, (TT_MUL, TT_DIV))
    
    def expr(self): # parse an expression composed of terms.
        res = ParseResult()

        if self.current_token.matches(TT_KWORD, 'let'):
            
            res.register_advancement()
            self.advance()

            if self.current_token.type != TT_VARIDENTIFIER:
                return res.failure(InvalidSyntaxError(self.current_token.pos_start, self.current_token.pos_end, "Expected Identifier"))
            
            var_name = self.current_token
            res.register_advancement()
            self.advance()

            if self.current_token.type == TT_EQ: 
                res.register_advancement()
                self.advance()
                expr = res.register(self.expr())
                if res.error: return res

                # Check if the expression is an IFNode and enforce else clause, like python
                if isinstance(expr, IFNode) and not expr.else_case:
                    return res.failure(InvalidSyntaxError(
                    expr.pos_start, expr.pos_end,
                    "Conditional expression in assignment requires an 'else' clause"
                ))

                return res.success(VarAssignNode(var_name, var_value=expr))
            

            elif self.current_token.type == TT_VARTYPEASSIGN:
                res.register_advancement()
                self.advance()
                if self.current_token.type != TT_VARTYPE : 
                    return res.failure(InvalidSyntaxError(self.current_token.pos_start, self.current_token.pos_end, "Expected 'for', 'if','let', Int, Float, Identifier, '+', '-', '*', '/', '[' or '('"))
                
                var_type = self.current_token
                res.register_advancement()
                self.advance()
                
                if self.current_token.type == TT_EQ: 
                    res.register_advancement()
                    self.advance()
                    expr = res.register(self.expr())
                    if res.error: return res
                    return res.success(VarAssignNode(var_name, var_type=var_type, var_value=expr))
                else:
                    return res.success(VarAssignNode(var_name, var_type=var_type))

            else:
                return res.failure(InvalidSyntaxError(self.current_token.pos_start, self.current_token.pos_end, "Expected variable a assign or '='"))


        else:
            # Trying to parse the left side of a potential assignment?

            node = res.register(self.bin_opr(self.comp_expr, ((TT_KWORD, 'and'), (TT_KWORD, 'or'))))
            if res.error: 
                return res.failure(InvalidSyntaxError(
                    self.current_token.pos_start, self.current_token.pos_end, 
                    "Expected 'for', 'if','let', Int, Float, Identifier, '+', '-', '*', '/', '[' or '('"
                ))
            
            # Check if this is an assignment ; node is VarAccessNode followed by '='

            if isinstance(node, VarAccessNode) and self.current_token.type == TT_EQ:
                var_name = node.var_name
                res.register_advancement()
                self.advance()
                

                expr = res.register(self.expr())
                

                if res.error: return res
            
                # Get variable info from symbol table, from the *future*
                var_info = self.future.get(var_name.value, None) if self.future else None
                
                if not var_info:
                    return res.failure(InvalidSyntaxError(
                        node.pos_start, node.pos_end,
                        f"Variable '{var_name.value}' is not defined"
                    ))

                return res.success(VarAssignNode(var_name, var_value=expr))
            
            return res.success(node)
    

    def list_expr(self):  # a list expr, parse a list creation LSPAREN value,* RSPAREN
        res = ParseResult() 
        element_nodes = []
        pos_start = self.current_token.pos_start.copy()

        if self.current_token.type != TT_LSPAREN:
            return res.failure(InvalidSyntaxError(
                    self.current_token.pos_start, self.current_token.pos_end,
                f"Expected '['"
            ))

        res.register_advancement()
        self.advance()

        if self.current_token.type == TT_RSPAREN:
            res.register_advancement()
            self.advance()

        else:
            element_nodes.append(res.register(self.expr()))
            if res.error: 
                return res.failure(InvalidSyntaxError(
                self.current_token.pos_start, self.current_token.pos_end, 
                "Expected ']', 'for', 'if', 'let', Int, Float, Identifier, '+', '-', '*', '/', '(', '[' or ')'"
            ))
            
            while self.current_token.type == TT_COMMA:
                res.register_advancement()
                self.advance()

                element_nodes.append(res.register(self.expr()))
                if res.error: return res

            if self.current_token.type != TT_RSPAREN:
                return res.failure(InvalidSyntaxError(
                    self.current_token.pos_start, self.current_token.pos_end,
                                        "Expected ',' or ']' "))     
            
            res.register_advancement()
            self.advance()

        return res.success(ListNode(element_nodes, pos_start, self.current_token.pos_end.copy()))

    def if_expr(self): # if expr, parse an if statement
        res = ParseResult() 
        all_cases = res.register(self.if_expr_case('if'))
        if res.error: return res 

        cases, else_case = all_cases
        return res.success(IFNode(cases, else_case))
    
    def elif_expr(self) :# elif expr, parse an elif statement, just does the same thing as if
        return self.if_expr_case('elif')
    
    def else_expr(self): # # else expr, parse an else statement, if any
        res = ParseResult() 
        else_case = None

        if self.current_token.matches(TT_KWORD, 'else'):
            res.register_advancement()
            self.advance()


            if not self.current_token.type == TT_LCPAREN: 
                while self.current_token.type == TT_NEWLINE: 
                    res.register_advancement()
                    self.advance()

                if self.current_token.type != TT_LCPAREN: 
                    return res.failure(InvalidSyntaxError(
                            self.current_token.pos_start, self.current_token.pos_end,
                            "Expected '{' "
                        ))
            
            res.register_advancement()
            self.advance()

            if self.current_token.type == TT_NEWLINE:
                
                res.register_advancement()
                self.advance()

                statements = res.register(self.statements())
                if res.error: return res 
                else_case = (statements, True)
    
                if self.current_token.type != TT_RCPAREN: 
                    return res.failure(InvalidSyntaxError(
                            self.current_token.pos_start, self.current_token.pos_end,
                            "Expected '}' "
                        ))
            
                res.register_advancement()
                self.advance()
                
            else:
                expr = res.register(self.expr())
                if res.error: return res
                else_case = (expr, False)

                if not self.current_token.type == TT_RCPAREN: 
                    while self.current_token.type == TT_NEWLINE: 
                        res.register_advancement()
                        self.advance()

                    if self.current_token.type != TT_RCPAREN: 
                        return res.failure(InvalidSyntaxError(
                                self.current_token.pos_start, self.current_token.pos_end,
                                "Expected '}' "
                            ))
            
                res.register_advancement()
                self.advance()

        return res.success(else_case)

    def elif_else_expr(self): # Elif/Else expr
        res = ParseResult() 
        cases = []
        else_case = None

        # Consume any newlines 

        while self.current_token.type == TT_NEWLINE:
            res.register_advancement()
            self.advance()

        if self.current_token.matches(TT_KWORD, 'elif'):
            all_cases = res.register(self.elif_expr())
            if res.error: return res
            cases, else_case = all_cases
        else: 
            else_case = res.register(self.else_expr())
            if res.error: return res

        return res.success((cases, else_case))


    def if_expr_case(self, case_kword): # Base if expr
        res = ParseResult() 
        cases = []
        else_case = None

        if not self.current_token.matches(TT_KWORD, case_kword):
            return res.failure(InvalidSyntaxError(
                        self.current_token.pos_start, self.current_token.pos_end,
                        f"Expected '{case_kword}' keyword"
                    ))
                    
        res.register_advancement()
        self.advance()

        condition = res.register(self.expr())
        if res.error : return res

        if not self.current_token.type == TT_LCPAREN: 
            while self.current_token.type == TT_NEWLINE: 
                res.register_advancement()
                self.advance()

            if self.current_token.type != TT_LCPAREN: 
                return res.failure(InvalidSyntaxError(
                        self.current_token.pos_start, self.current_token.pos_end,
                        "Expected '{' "
                    ))
        
        res.register_advancement()
        self.advance()
    
        if self.current_token.type == TT_NEWLINE:
            res.register_advancement()
            self.advance()

            statements = res.register(self.statements())
            
            if res.error: return res 
            cases.append((condition, statements, True))
    
            while self.current_token.type == TT_NEWLINE:
                res.register_advancement()
                self.advance()

            if self.current_token.type != TT_RCPAREN:
                return res.failure(InvalidSyntaxError(
                    self.current_token.pos_start, self.current_token.pos_end,
                    "Expected '}'"
                ))
            res.register_advancement()
            self.advance()

            # Consume new lines, yum 
            while self.current_token.type == TT_NEWLINE:
                res.register_advancement()
                self.advance()

            all_cases = res.register(self.elif_else_expr())
            if res.error: return res
            new_cases, else_case = all_cases
            cases.extend(new_cases)
        else:
            expr = res.register(self.expr())
            if res.error: return res
            cases.append((condition, expr, False))

            
            if not self.current_token.type == TT_RCPAREN: 
                while self.current_token.type == TT_NEWLINE: 
                    res.register_advancement()
                    self.advance()

            if self.current_token.type != TT_RCPAREN: 
                return res.failure(InvalidSyntaxError(
                        self.current_token.pos_start, self.current_token.pos_end,
                        "Expected '}' "
                    ))
        
            res.register_advancement()
            self.advance()

            all_cases = res.register(self.elif_else_expr())
            if res.error: return res
            new_cases, else_case = all_cases
            cases.extend(new_cases)

        
        return res.success((cases, else_case))

    def for_expr(self): # for loop expr 
        res = ParseResult()

        if not self.current_token.matches(TT_KWORD, 'for'):
            return res.failure(InvalidSyntaxError(
                self.current_token.pos_start, self.current_token.pos_end,
                f"Expected 'for'"
            ))

        res.register_advancement()
        self.advance()

        if self.current_token.type != TT_VARIDENTIFIER:
            return res.failure(InvalidSyntaxError(
                self.current_token.pos_start, self.current_token.pos_end,
                f"Expected identifier"
            ))

        var_name = self.current_token
        res.register_advancement()
        self.advance()

        if not self.current_token.matches(TT_KWORD, 'in'):
            return res.failure(InvalidSyntaxError(
                self.current_token.pos_start, self.current_token.pos_end,
                f"Expected 'in'"
            ))

        res.register_advancement()
        self.advance()

        start_value = res.register(self.expr())
        
        if res.error: return res
        
        if not self.current_token != TT_RANGE:
            
            return res.failure(InvalidSyntaxError(
                self.current_token.pos_start, self.current_token.pos_end,
                f"Expected Range Identifier '..'"
            ))

        res.register_advancement()
        self.advance()

        end_value = res.register(self.expr())
        

        if res.error: return res

        step_value = None

        if self.current_token.type == TT_RANGE:
            res.register_advancement()
            self.advance()

            step_value = res.register(self.expr())
            
            
            if res.error: return res

        if not self.current_token.type == TT_LCPAREN: 
            while self.current_token.type == TT_NEWLINE: 
                res.register_advancement()
                self.advance()

            if self.current_token.type != TT_LCPAREN: 
                return res.failure(InvalidSyntaxError(
                        self.current_token.pos_start, self.current_token.pos_end,
                        "Expected '{' "
                    ))

        res.register_advancement()
        self.advance()

        if self.current_token.type == TT_NEWLINE:
            res.register_advancement()
            self.advance()

            body = res.register(self.statements())
            if res.error: return res 

            
            if self.current_token.type != TT_RCPAREN: 
                return res.failure(InvalidSyntaxError(
                        self.current_token.pos_start, self.current_token.pos_end,
                        "Expected '}' "
                    ))
            
                
            res.register_advancement()
            self.advance()

            return res.success(ForNode(var_name, start_value, end_value, step_value, body ,True))
            
        else:
            body = res.register(self.expr())
            if res.error: return res
            if  self.current_token.type != TT_RCPAREN: 
                return res.failure(InvalidSyntaxError(
                            self.current_token.pos_start, self.current_token.pos_end,
                            "Expected '}' "
                        ))
                
            res.register_advancement()
            self.advance()
            
            return res.success(ForNode(var_name, start_value, end_value, step_value, body, False))

    def while_expr(self) :# while loop expr
        res = ParseResult()

        if not self.current_token.matches(TT_KWORD, 'while'):
            return res.failure(InvalidSyntaxError(
                self.current_token.pos_start, self.current_token.pos_end,
                f"Expected 'while'"
            ))

        res.register_advancement()
        self.advance()

        condition = res.register(self.expr())
        if res.error: return res

        if not self.current_token.type == TT_LCPAREN: 
            while self.current_token.type == TT_NEWLINE: 
                res.register_advancement()
                self.advance()

            if self.current_token.type != TT_LCPAREN: 
                return res.failure(InvalidSyntaxError(
                        self.current_token.pos_start, self.current_token.pos_end,
                        "Expected '{' "
                    ))

        res.register_advancement()
        self.advance()

        if self.current_token.type == TT_NEWLINE:
            res.register_advancement()
            self.advance()

            body = res.register(self.statements())
            if res.error: return res 

            
            if self.current_token.type != TT_RCPAREN: 
                return res.failure(InvalidSyntaxError(
                        self.current_token.pos_start, self.current_token.pos_end,
                        "Expected '}' "
                    ))
            
                
            res.register_advancement()
            self.advance()
            return res.success(WhileNode(condition, body, True))

        else:
            body = res.register(self.expr())
            if res.error: return res

            if not self.current_token.type == TT_RCPAREN: 
                return res.failure(InvalidSyntaxError(
                            self.current_token.pos_start, self.current_token.pos_end,
                            "Expected '}' "
                        ))
                
            res.register_advancement()
            self.advance()

            return res.success(WhileNode(condition, body, False))

    def fun_def(self): # parse a function definition 
        res = ParseResult()

        if not self.current_token.matches(TT_KWORD, 'fun'):
            return res.failure(InvalidSyntaxError(
                        self.current_token.pos_start, self.current_token.pos_end,
                        "Expected keyword 'fun' "
                    ))

        res.register_advancement()
        self.advance()

        if not self.current_token.type == TT_VARIDENTIFIER: 
            return res.failure(InvalidSyntaxError(
                        self.current_token.pos_start, self.current_token.pos_end,
                        "Expected Identifier "
                    ))

        var_name_token = self.current_token

        res.register_advancement()
        self.advance()

        if not self.current_token.type == TT_LPAREN: 
            return res.failure(InvalidSyntaxError(
                        self.current_token.pos_start, self.current_token.pos_end,
                        "Expected '(' "
                    ))

        res.register_advancement()
        self.advance()

        arg_name_tokens = []
        if self.current_token.type == TT_VARIDENTIFIER:
            arg_name_tokens.append(self.current_token )

            res.register_advancement()
            self.advance()

            while self.current_token.type == TT_COMMA:
                res.register_advancement()
                self.advance()
                if self.current_token.type != TT_VARIDENTIFIER:
                    return res.failure(InvalidSyntaxError(
                        self.current_token.pos_start, self.current_token.pos_end,
                        "Expected identifier "
                    ))

                arg_name_tokens.append(self.current_token )
                res.register_advancement()
                self.advance()
            
            if self.current_token.type != TT_RPAREN:
                return res.failure(InvalidSyntaxError(
                        self.current_token.pos_start, self.current_token.pos_end,
                                            "Expected ',' or ')' "))      
        else:
            if self.current_token.type != TT_RPAREN:
                return res.failure(InvalidSyntaxError(
                        self.current_token.pos_start, self.current_token.pos_end,
                                            "Expected ',' or ')' "))      

        res.register_advancement()
        self.advance()

        if not self.current_token.type == TT_LCPAREN: 
            while self.current_token.type == TT_NEWLINE: 
                res.register_advancement()
                self.advance()

            if self.current_token.type != TT_LCPAREN: 
                return res.failure(InvalidSyntaxError(
                        self.current_token.pos_start, self.current_token.pos_end,
                        "Expected '{' "
                    ))

        res.register_advancement()
        self.advance()

        if self.current_token.type == TT_NEWLINE:
            res.register_advancement()
            self.advance()

            body = res.register(self.statements())
            if res.error: return res

            if self.current_token.type != TT_RCPAREN: 
                return res.failure(InvalidSyntaxError(
                            self.current_token.pos_start, self.current_token.pos_end,
                            "Expected '}' "
                        ))
            res.register_advancement()
            self.advance()

            return res.success(FunDefNode(var_name_token, arg_name_tokens, body, True))
        else:

            body = res.register(self.expr())
            if res.error: return res

            if self.current_token.type != TT_RCPAREN: 
                return res.failure(InvalidSyntaxError(
                            self.current_token.pos_start, self.current_token.pos_end,
                            "Expected '}' "
                        ))
            
            res.register_advancement()
            self.advance()

            return res.success(FunDefNode(var_name_token, arg_name_tokens, body, False))

    
    def anonymous_fun_def(self):  # parse an anonymous function definition (a tilda ~) 
        res = ParseResult()

        if not self.current_token.type == TT_TILDA:
            return res.failure(InvalidSyntaxError(
                        self.current_token.pos_start, self.current_token.pos_end,
                        "Expected '~' for anonymous function definition"
                    ))
        
        res.register_advancement()
        self.advance()

        var_name_token = None
        arg_name_tokens = []

        if self.current_token.type == TT_VARIDENTIFIER:
            arg_name_tokens.append(self.current_token )

            res.register_advancement()
            self.advance()

            while self.current_token.type == TT_COMMA:
                res.register_advancement()
                self.advance()

                if self.current_token.type != TT_VARIDENTIFIER:
                    return res.failure(InvalidSyntaxError(
                        self.current_token.pos_start, self.current_token.pos_end,
                        "Expected identifier "
                    ))

                arg_name_tokens.append(self.current_token )

                res.register_advancement()
                self.advance()
            
                
        else:
            return res.failure(InvalidSyntaxError(
                self.current_token.pos_start, self.current_token.pos_end,
                                            "Expected Identifier(s) "))      

        if self.current_token.type != TT_LCPAREN: 
            return res.failure(InvalidSyntaxError(
                    self.current_token.pos_start, self.current_token.pos_end,
                        "Expected '{' "
                    ))

        res.register_advancement()
        self.advance()

        body = res.register(self.expr())
        if res.error: return res

        if self.current_token.type != TT_RCPAREN: 
            return res.failure(InvalidSyntaxError(
                        self.current_token.pos_start, self.current_token.pos_end,
                        "Expected '}' "
                    ))
        
        res.register_advancement()
        self.advance()

        return res.success(FunDefNode(var_name_token, arg_name_tokens, body))

    def comp_expr(self):
        res = ParseResult()
        
        if self.current_token.matches(TT_KWORD, 'not'): # Looking for NOT, ofcourse
            opr_token =  self.current_token
            res.register_advancement()
            self.advance()

            node = res.register(self.comp_expr())
            if res.error: return res 
            return res.success(UnaryOpNode(opr_token, node))

        # But if NOT is not found, then we will look for rule 2: Arith-Expr ((EE|NE|LT|GT|LTE|GTE) Arith-Expr)*
        node  = res.register(self.bin_opr(self.shift_expr, (TT_EE, TT_NE, TT_LT, TT_GT, TT_LTE, TT_GTE)))
        if res.error: 
            return res.failure(InvalidSyntaxError(self.current_token.pos_start, self.current_token.pos_end, "Expected Int, Float, Identifier, '+', '-', '*', '/', '[', '(' or 'not' "))

        return res.success(node)

    def arith_expr(self): # Arithmetic expr
        return self.bin_opr(self.term, (TT_PLUS, TT_MINUS))
    
    def shift_expr(self): # List Shifting expr
        return self.bin_opr(self.arith_expr, (TT_LSHIFT, TT_RSHIFT))
    
    def bin_opr(self, func_a, accepted_oprs, func_b=None):
        if func_b == None:
            func_b = func_a

        res = ParseResult()
        left = res.register(func_a())
        
        if res.error: return res

        while (self.current_token.type, self.current_token.value) in accepted_oprs or self.current_token.type in accepted_oprs:
            opr_token = self.current_token
            res.register_advancement()
            self.advance()
            right = res.register(func_b())
            if res.error: return res
            left = BinOpNode(left, opr_token, right)

        return res.success(left)


# Runtime result class, almost as same as ParseResult 
class RuntimeResult:
    def __init__(self):
        self.error = None
        self.value = None
    
    def  success(self, value):
        self.value = value 
        return self
    
    def register(self, res):
        if res.error: self.error = res.error
        return res.value
        
    def failure(self, error):
        self.error = error
        return self

# Runtime Value representation classes, Value is the base class.
class Value:
	def __init__(self):
		self.set_pos()
		self.set_context()

	def set_pos(self, pos_start=None, pos_end=None):
		self.pos_start = pos_start
		self.pos_end = pos_end
		return self

	def set_context(self, context=None):
		self.context = context
		return self

	def add(self, other):
		return None, self.illegal_operation(other)

	def subtract(self, other):
		return None, self.illegal_operation(other)

	def multiply(self, other):
		return None, self.illegal_operation(other)

	def divide(self, other):
		return None, self.illegal_operation(other)

	def raised_to(self, other):
		return None, self.illegal_operation(other)

	def get_comparison_eq(self, other):
		return None, self.illegal_operation(other)

	def get_comparison_ne(self, other):
		return None, self.illegal_operation(other)

	def get_comparison_lt(self, other):
		return None, self.illegal_operation(other)

	def get_comparison_gt(self, other):
		return None, self.illegal_operation(other)

	def get_comparison_lte(self, other):
		return None, self.illegal_operation(other)

	def get_comparison_gte(self, other):
		return None, self.illegal_operation(other)

	def and_(self, other):
		return None, self.illegal_operation(other, details=f"Operator 'and' expects Bool operands, got {type(self).__name__} and {type(other).__name__}")

	def or_(self, other):
		return None, self.illegal_operation(other, details=f"Operator 'or' expects Bool operands, got {type(self).__name__} and {type(other).__name__}")

	def not_(self):
		return None, self.illegal_operation(details=f"Operator 'not' expects Bool operand, got {type(self).__name__}")

	def execute(self, args):
		return RuntimeResult().failure(self.illegal_operation())

	def copy(self):
		raise Exception('No copy method defined')

	def is_true(self):
		return False

	def illegal_operation(self, details=None, other=None):
		if not other: other = self
		return RuntimeError(
			self.pos_start, other.pos_end,
			f"Illegal operation{f' : {details}' if details else ''}",
			self.context
		)

# Void Runtime Value representation 
class Void(Value):
    def __init__(self):
        super().__init__()
        self.value = None

    def is_true(self):
        return False  
    
    # def add(self, other):
    #     return None, RuntimeError(self.pos_start, other.pos_end, "Cannot operate on Void", self.context)

    def copy(self):
        copy = Void()
        copy.set_pos(self.pos_start, self.pos_end)
        copy.set_context(self.context)
        return copy

    def __repr__(self):
        return "Null"

# Bool Runtime Value representation 
class Bool(Value):
    def __init__(self, value):
        super().__init__()

        if isinstance(value, int):
            self.value = True if value == 1 else False

        elif isinstance(value, Number):
            self.value = True if value.value == 1 else False
        elif isinstance(value, Bool):
            self.value = True if value.value == True else False

        elif value in ('True', 'False'):
            self.value = True if value == 'True' else False
        
        self.context = None
        self.set_pos()

    def and_(self, other):
        if isinstance(other, Bool):
            return Bool(self.value and other.value).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self, other)           
        
    def or_(self, other):
        if isinstance(other, Bool):
            return Bool(self.value or other.value).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self, other)           
        
    def not_(self):
        return Bool(not self.value).set_context(self.context), None
    
    def is_true(self):
        return self.value == True
    
    def copy(self):
        copy = Bool(self.value)
        copy.set_pos(self.pos_start, self.pos_end)
        copy.set_context(self.context)

        return copy

    def __repr__(self):
        return str(self.value)
    
# Number Runtime Value representation 
class Number(Value):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def add(self, other):
        if isinstance(other, Number):
            return Number(self.value + other.value).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self, other)           
        
    def subtract(self, other):
        if isinstance(other, Number):
            return Number(self.value - other.value).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self, other)           
        
    def multiply(self, other):
        if isinstance(other, Number):
            return Number(self.value * other.value).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self, other)           

    def divide(self, other):
        if isinstance(other, Number):
            if other.value == 0:
        
                return None, RuntimeError(other.pos_start, other.pos_end, 'Division by Zero', self.context)
            return Number(self.value / other.value).set_context(self.context), None
        

        else:
            return None, Value.illegal_operation(self, other)           
            
    def raise_to(self, other):
        if isinstance(other, Number):
            return Number(self.value ** other.value).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self, other)           
        
    def get_comparison_eq(self, other):
        if isinstance(other, Number):
            return Number(int(self.value == other.value)).set_context(self.context), None
        
        else:
            return None, Value.illegal_operation(self, other)           
        

    def get_comparison_ne(self, other):
        if isinstance(other, Number):
            return Number(int(self.value != other.value)).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self, other)           
        
    def get_comparison_lt(self, other):
        if isinstance(other, Number):
            return Number(int(self.value < other.value)).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self, other)           
        
    def get_comparison_gt(self, other):
        if isinstance(other, Number):
            return Number(int(self.value > other.value)).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self, other)           
        
    def get_comparison_lte(self, other):
        if isinstance(other, Number):
            return Number(int(self.value <= other.value)).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self, other)           

    def get_comparison_gte(self, other):
        if isinstance(other, Number):
            return Number(int(self.value >= other.value)).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self, other)           
    
    def is_true(self):
        return self.value != 0 

    def copy(self):
        copy = Number(self.value)
        copy.set_pos(self.pos_start, self.pos_end)
        copy.set_context(self.context)

        return copy

    def __repr__(self):
        return str(self.value)

# String Runtime Value representation 
class String(Value): 
    def __init__(self, value):
        super().__init__()      
        self.value = str(value)
        self.set_pos()
        self.set_context()

    def add(self, other):
        if isinstance(other, String):
            return String(self.value + other.value ).set_context(self.context), None
        else:
            return None, Value.illegal_operation(self, other, details=f"Object of type '{type(other)}' cannot be added to a String")           
    
    def multiply(self, other):
        if isinstance(other, Number):
            return String(self.value * other.value ).set_context(self.context), None
        
        else:
            return None, Value.illegal_operation(self, other, details=f"Object of type '{type(other)}' cannot be multiplied to a String")           
    
   
    def subtract(self, other):
        return None, Value.illegal_operation(self, other, details=f"Unsupported operator '-' for type String")           

    def divide(self, other):
        return None, Value.illegal_operation(self, other, details=f"Unsupported operator '/' for type String")           

    def raise_to(self, other):
        return None, Value.illegal_operation(self, other, details=f"Unsupported operator '**' for type String")           

    def access(self, other): # Access a char
        if isinstance(other, Number):
            idx = int(other.value)
            
            if idx < 0:
                idx = len(self.value) + idx
            
            try:
                char = self.value[idx]
                return String(char).set_context(self.context), None
            
            except IndexError:
                return None, RuntimeError(
                    other.pos_start, other.pos_end,
                    f"Cannot access character at index {other.value}, Index is out of bounds",
                    self.context
                )
            
        # Handle range 
        elif isinstance(other, Range):
            start = other.start
            end = other.end
            step = other.step

            # Validate start and end types
            if (start is not None and not isinstance(start, Number)) or (end is not None and not isinstance(end, Number)):
                return None, Value.illegal_operation(self, details=f"Range indices must be Numbers, got {type(start).__name__} and {type(end).__name__}")
            
            # Validate step 
            if step is not None and not isinstance(step, Number):
                return None, Value.illegal_operation(self, details=f"Step must be a Number, got {type(step).__name__}")
            
            # Default values for omitted bounds
            start_idx = 0 if start is None else int(start.value)
            end_idx = len(self.value) if end is None else int(end.value)
            step_value = 1 if step is None else int(step.value)

            
            if start_idx < 0:
                start_idx = len(self.value) + start_idx
            if end_idx < 0:
                end_idx = len(self.value) + end_idx

            
            start_idx = max(0, min(start_idx, len(self.value)))
            end_idx = max(0, min(end_idx, len(self.value)))

            
            if step_value != 0:  # Avoid division by zero

                if step_value < 0:
                    # Reverse slicing: swap start and end, negate step

                    start_idx, end_idx = end_idx, start_idx
                    step_value = -step_value
                
                if start_idx > end_idx:
                    # Swap to ensure correct slicing direction
                    start_idx, end_idx = end_idx, start_idx

            # Perform slicing with step
            try:
                if step_value == 0:
                    return None, RuntimeError(other.pos_start, other.pos_end, "Step cannot be zero", self.context)
                
                # Create a list of indices based on step
                indices = range(start_idx, end_idx, step_value)
                sliced_chars = [self.value[i] for i in indices]
                sliced_str = "".join(sliced_chars)
                return String(sliced_str).set_context(self.context), None
            except Exception as e:
                return None, RuntimeError(other.pos_start, other.pos_end, f"Error slicing string: {str(e)}", self.context)

        else:
            
            return None, Value.illegal_operation(self, details=f"Index must be a Number or Range, got {type(other).__name__}")

        
    def get_comparison_eq(self, other):
        return Bool(self.value == other.value).set_context(self.context), None
        
    def get_comparison_ne(self, other):
        return Bool(self.value == other.value).set_context(self.context), None

    def get_comparison_lt(self, other):
        return None, Value.illegal_operation(self, other, details=f"Unsupported operator '<' for type String")           
    def get_comparison_gt(self, other):
        return None, Value.illegal_operation(self, other, details=f"Unsupported operator '>' for type String")           
    def get_comparison_lte(self, other):
        return None, Value.illegal_operation(self, other, details=f"Unsupported operator '<=' for type String")           
    def get_comparison_gte(self, other):
        return None, Value.illegal_operation(self, other, details=f"Unsupported operator '>=' for type String")           
    def and_(self, other):
        return None, Value.illegal_operation(self, other, details=f"Operator 'and' expects Bool operands, got String and {type(other).__name__}")           
    
    def or_(self, other):
        return None, Value.illegal_operation(self, other, details=f"Operator 'or' expects Bool operands, got String and {type(other).__name__}")           

    def not_(self):
        return None, Value.illegal_operation(self, details=f"Operator 'not' expects Bool, got String")           

    def is_true(self):
        return len(self.value) > 0 
    
    def set_context(self, context=None):
        self.context = context 
        return self
    
    def set_pos(self, pos_start=None, pos_end=None):
        self.pos_start = pos_start
        self.pos_end = pos_end
        return self
    
    def copy(self):
        copy = String(self.value)
        copy.set_pos(self.pos_start, self.pos_end)
        copy.set_context(self.context)

        return copy
    
    def __repr__(self):
        return f'"{self.value}"'

# List Runtime Value representation 
class List(Value):
    def __init__(self, elements):
        super().__init__()
        self.elements = elements


    def add(self, other):
        if isinstance(other, List):
            new_list = self.copy() #Immutable
            new_list.elements.extend(other.elements) 
            return new_list, None
        else:
            return None, Value.illegal_operation(self, details=f"Operator '+' for lists expects a list operand, got {type(other).__name__}")           

    def append(self, other): #LSHIFT

        if isinstance(other, List):
            new_list = self.copy() #Immutable
            new_list.elements.extend(other.elements) 
            return new_list, None
        else:
            new_list = self.copy() #Immutable
            new_list.elements.append(other.value) 
            return new_list, None
    
    def remove(self, other): #RSHIFT
        if isinstance(other, Number):
            new_list = self.copy() #Immutable
            try:
                new_list.elements.pop(other.value)
                return new_list, None
            except:
                return None, RuntimeError(other.pos_start, other.pos_end, f'Cannot remove element at index {other.value}, Index is out of bounds', self.context)           

        else:
            return None, Value.illegal_operation(self, details=f"Operator '>>' expects a Number operand as index, got {type(other).__name__}")           

    def access(self, other):
        if isinstance(other, Number):

            idx = int(other.value)
            if idx < 0: # for -ve index
                idx = len(self.elements) + idx

            try:
                return self.elements[other.value], None
            except:
                return None, RuntimeError(other.pos_start, other.pos_end, f'Cannot access element at index {other.value}, Index is out of bounds')           
        
        # Handle range index
        elif isinstance(other, Range):
            start = other.start
            end = other.end
            step = other.step

            # Validate start and end types
            if (start is not None and not isinstance(start, Number)) or (end is not None and not isinstance(end, Number)):
                return None, Value.illegal_operation(self, details=f"Range indices must be Numbers, got {type(start).__name__} and {type(end).__name__}")
            
            # Validate step type
            if step is not None and not isinstance(step, Number):
                return None, Value.illegal_operation(self, details=f"Step must be a Number, got {type(step).__name__}")

            # Default values for omitted bounds
            start_idx = 0 if start is None else int(start.value)
            end_idx = len(self.elements) if end is None else int(end.value)
            step_value = 1 if step is None else int(step.value)

            
            if start_idx < 0:
                start_idx = len(self.elements) + start_idx
            if end_idx < 0:
                end_idx = len(self.elements) + end_idx

            start_idx = max(0, min(start_idx, len(self.elements)))
            end_idx = max(0, min(end_idx, len(self.elements)))

            if step_value != 0:  
                if step_value < 0:
                    start_idx, end_idx = end_idx, start_idx
                    step_value = -step_value

                if start_idx > end_idx:
                    start_idx, end_idx = end_idx, start_idx

            try:
                if step_value == 0:
                    return None, RuntimeError(other.pos_start, other.pos_end, "Step cannot be zero", self.context)
                
                # New sliced list
                indices = range(start_idx, end_idx, step_value)
                sliced_elements = [self.elements[i] for i in indices]
                return List(sliced_elements).set_context(self.context), None
            
            except Exception as e:
                return None, RuntimeError(other.pos_start, other.pos_end, f"Error slicing list: {str(e)}", self.context)

        else:
            return None, Value.illegal_operation(self, details=f"Index must be a Number or Range, got {type(other).__name__}")
        
    def copy(self):
        copy = List(self.elements[:])
        copy.set_pos(self.pos_start, self.pos_end)
        copy.set_context(self.context)
        return copy
    
    def __repr__(self):
        return f"[{', '.join([str(ele) for ele in self.elements])}]"
    
# Range representation 
class Range:
    def __init__(self, start, end, step, pos_start, pos_end):
        self.start = start  # Start index (a Number or None)
        self.end = end      # End index (a Number or None)
        self.step = step    # Step value (a Number or None)
        self.pos_start = pos_start
        self.pos_end = pos_end

# Base Function representation 
class BaseFunction(Value):
    def __init__(self, name):
        super().__init__()
        self.name= name or '<anonymous-function>'

    def generate_new_ctx(self):
        ctx = Context(self.name, self.context, self.pos_start)
        ctx.symbol_table = SymbolTable(ctx.parent.symbol_table)
        return ctx
    
    def check_args(self, arg_names, args):
        rt_res = RuntimeResult()

        if len(args) > len(arg_names):
            return rt_res.failure(RuntimeError(self.pos_start, self.pos_end, f"Too Many Arguments Passed : {self.name} takes {len(arg_names)} arguments but {len(args)} were given", self.context))

        
        if len(args) < len(arg_names):
            return rt_res.failure(RuntimeError(self.pos_start, self.pos_end, f"Too Few Arguments Passed: {self.name} takes {len(arg_names)} arguments but {len(args)} were given", self.context))
        
        return rt_res.success(None)
    
    def populate_args(self, arg_names, args, ctx):
        for i, arg in enumerate(args):
            arg_name = arg_names[i]
            arg_value = arg
            
            arg_value.set_context(ctx)
            ctx.symbol_table.set(arg_name, None, arg_value) 

    def check_populate_args(self, arg_names, args, ctx):
        rt_res = RuntimeResult()

        rt_res.register(self.check_args(arg_names, args))
        if rt_res.error: return rt_res
        
        self.populate_args(arg_names, args, ctx)

        return rt_res.success(None)
        
# User-Defined Function representation 
class Function(BaseFunction):
    def __init__(self, name, body_node, arg_names, retr_void):
        super().__init__(name)
        self.body_node = body_node
        self.arg_names = arg_names
        self.retr_void = retr_void

    def execute(self, args):
        rt_res = RuntimeResult()
        intp = Interpreter()
        ctx = self.generate_new_ctx() # Execution context
        
        rt_res.register(self.check_populate_args(self.arg_names, args, ctx))
        if rt_res.error: return rt_res 

        value = rt_res.register(intp.visit(self.body_node, ctx))
        if rt_res.error: return rt_res 
        
        return rt_res.success(Void() if self.retr_void else value)

    def copy(self):
        copy = Function(self.name, self.body_node, self.arg_names, self.retr_void) 
        copy.set_pos(self.pos_start, self.pos_end)
        copy.set_context(self.context)

        return copy
    
    def __repr__(self):
        return f'<function {self.name}>'
    
# Built-In Function representation 
class BuiltInFunction(BaseFunction):
    def __init__(self, name):
        super().__init__(name) 

    def execute(self, args):
        rt_res = RuntimeResult()
        ctx = self.generate_new_ctx() # Execution context
        
        method_name = f'exec_{self.name}'
        method = getattr(self, method_name, self.no_visit_method)

        rt_res.register(self.check_populate_args(method.arg_names, args, ctx))
        if rt_res.error: return rt_res 

        return_value = rt_res.register(method(ctx))
        if rt_res.error: return rt_res 
    
        return rt_res.success(return_value)
    
    def no_visit_method(self):
        raise Exception(f'No execute_{self.name} method defined.')
    
    def copy(self):
        copy = BuiltInFunction(self.name)
        copy.set_pos(self.pos_start, self.pos_end)
        copy.set_context(self.context)
        return copy
    
    def __repr__(self):
        return f'<built-in function {self.name}'

    # Built-In Methods:
    # Print
    def exec_print(self, exec_ctx):
        _, value = exec_ctx.symbol_table.get('value')
        if value is None:
            return RuntimeResult().failure(RuntimeError(
                self.pos_start, self.pos_end,
                "Argument 'value' is undefined",
                exec_ctx
            ))

        if isinstance(value, Number) or isinstance(value, Bool) or isinstance(value, String):
            str_value = str(value.value)
        elif isinstance(value, Void):
            str_value = "Null"
        else:
            str_value = str(value)

        sys.stdout.write(str_value)
        sys.stdout.write('\n')
        sys.stdout.flush()

        return RuntimeResult().success(Void())

    exec_print.arg_names = ['value']

    def exec_return(self, exec_ctx):
        _, value = exec_ctx.symbol_table.get('value')
        if value is None:
            return RuntimeResult().failure(RuntimeError(
                self.pos_start, self.pos_end,
                "Argument 'value' is undefined",
                exec_ctx
            ))

        if isinstance(value, Number) or isinstance(value, Bool) or isinstance(value, String):
            str_value = value.value
        elif isinstance(value, Void):
            str_value = "Null"

        else:
            str_value = str(value)

        return RuntimeResult().success(String(str_value))
    
    exec_return.arg_names = ['value']
    
    # Input
    def exec_input(self, exec_ctx):
        _, prompt = exec_ctx.symbol_table.get('prompt')
        if prompt:
            if isinstance(prompt, Number):
                prompt_str = str(prompt.value)
            elif isinstance(prompt, Bool):
                prompt_str = str(prompt.value).lower()
            elif isinstance(prompt, String):
                prompt_str = prompt.value
            elif isinstance(prompt, Void):
                prompt_str = "Null"

            else:
                prompt_str = str(prompt)
            sys.stdout.write(prompt_str)
            sys.stdout.flush()

        
        user_input = sys.stdin.readline()
        if not sys.stdin.readable():
            return RuntimeResult().failure(RuntimeError(
            self.pos_start, self.pos_end,
            "Standard input is unreadable",
            exec_ctx
            ))
        
        user_input = user_input.rstrip('\n')

        return RuntimeResult().success(String(user_input).set_context(exec_ctx).set_pos(self.pos_start, self.pos_end))
    
    exec_input.arg_names = ['prompt']
    
    # Clear
    def exec_clear(self, exec_ctx):
        os.system('cls' if os.name == 'nt' else 'clear')
        return RuntimeResult().success(Void())
    
    exec_clear.arg_names = []

    # Run
    def exec_run(self, exec_ctx):
        _, fn = exec_ctx.symbol_table.get('filename')
        if not isinstance(fn, String):
            return RuntimeResult().failure(RuntimeError(
                        self.pos_start, self.pos_end,
                        "Argument 'filename' must be a string, got " + str(type(fn).__name__),
                        exec_ctx
                    ))

        fn = fn.value
        try:
            with open(fn, 'r') as f:
                script = f.read()
        except FileNotFoundError:
                return RuntimeResult().failure(RuntimeError(
                    self.pos_start, self.pos_end,
                    f"Cannot load script: File '{fn}' not found",
                    exec_ctx
                ))
        except PermissionError:
            return RuntimeResult().failure(RuntimeError(
                self.pos_start, self.pos_end,
                f"Cannot load script: Permission denied accessing '{fn}'",
                exec_ctx
            ))
        except IOError as e:
            return RuntimeResult().failure(RuntimeError(
                self.pos_start, self.pos_end,
                f"Cannot load script '{fn}': I/O error occurred ({str(e)})",
                exec_ctx
            ))
        except Exception as e:
            return RuntimeResult().failure(RuntimeError(
                self.pos_start, self.pos_end,
                f"Unexpected error loading script '{fn}': {str(e)}",
                exec_ctx
            ))

        _, error = run(fn, script)
        if error:
            return RuntimeResult().failure(RuntimeError(
                        self.pos_start, self.pos_end,
                        f"Execution of script '{fn}' failed:\n{error.as_string()}",
                        exec_ctx
                    ))
        return RuntimeResult().success(Void())

    exec_run.arg_names = ['filename']

    # Exit
    def exec_exit(self, exec_ctx):
        _, status = exec_ctx.symbol_table.get('status')
        if status is None:
            status_code = 0 
        elif isinstance(status, Number):
            status_code = int(status.value)
        else:
            return RuntimeResult().failure(RuntimeError(
                self.pos_start, self.pos_end,
                "Exit status must be a number",
                exec_ctx
            ))
        
        sys.exit(status_code)

    exec_exit.arg_names = []

# Assigining built ins to actual vars
BuiltInFunction.print  = BuiltInFunction('print')
BuiltInFunction.input  = BuiltInFunction('input')
BuiltInFunction.return_  = BuiltInFunction('return')
BuiltInFunction.clear  = BuiltInFunction('clear')
BuiltInFunction.run  = BuiltInFunction('run')
BuiltInFunction.exit  = BuiltInFunction('exit')

# CONTEXT : The context/focus can be code, a fucntion, etc
class Context:
    def __init__(self, display_name, parent=None, parent_entry_pos=None ):
        self.display_name = display_name
        self.parent = parent
        self.parent_entry_pos = parent_entry_pos
        self.symbol_table = None

# SYMBOL TABLE STORES PREDEFINED VARS, AND RUNTIME VARS
class SymbolTable:
    def __init__(self, parent=None):
        self.symbols = {}
        self.parent = parent

    def get(self, name):
        symbol = self.symbols.get(name, None)
        value = None
        type_ = None
        if symbol: 
            type_ = symbol[0] if symbol[0] else None
            value = symbol[1] if symbol[1] else None
            if type_ == None and value != None:
                type_ = type(value).__name__.capitalize()

        if value == None and self.parent: 
            return self.parent.get(name)
        
        return type_, value 

    def set(self, name, type_hint, value):
        if type_hint is None and value is not None:
            
            if isinstance(value, Number):
                inferred_type = 'Number'
            elif isinstance(value, String):
                inferred_type = 'String'
            elif isinstance(value, List):
                inferred_type = 'List'
            elif isinstance(value, Bool):
                inferred_type = 'Bool'
            elif isinstance(value, Void):
                inferred_type = 'Void'
            elif isinstance(value, Function):
                inferred_type = 'Function'
            else:
                # Fallback for unknown types
                inferred_type = type(value).__name__.capitalize()

        elif type_hint is not None:
            inferred_type = type_hint.value if hasattr(type_hint, 'value') else type_hint
        else:
            inferred_type = 'Void'
        self.symbols[name] = [inferred_type, value]

    def remove(self, name):
        del self.symbols[name]

# INTERPRETER : Evaluate parsed code.
class Interpreter:
    # Visit nodes in the AST to evaluate them
    def visit(self, node, context):
        method_name = f'visit_{type(node).__name__}'
        method = getattr(self, method_name, self.no_visit_method)
        return method(node, context)
    
    def numerize(self, bool):
        pos_end = bool.pos_start.copy()
        pos_end.idx += 1
        pos_end.colno += 1
        numnode = Number(bool.value).set_context(bool.context).set_pos(pos_start=bool.pos_start, pos_end=pos_end)
        return numnode

    # If the node is unknown
    def no_visit_method(self, node, context):
        raise Exception(f"No visit method named '{type(node).__name__}' defined.")
    
    # visit and eval for loop node and so on.. 
    def visit_ForNode(self, node, context):
        rt_res = RuntimeResult()
        elements = []
        start_value = rt_res.register(self.visit(node.start_value_node, context))
        if rt_res.error: return rt_res

        end_value = rt_res.register(self.visit(node.end_value_node, context))
        if rt_res.error: return rt_res

        step_value = Number(1)

        if node.step_value_node:
            step_value = rt_res.register(self.visit(node.step_value_node, context))
            if rt_res.error: return rt_res
        
        if not all(isinstance(v, Number) for v in [start_value, end_value, step_value]):
            return rt_res.failure(RuntimeError(
                node.pos_start, node.pos_end,
                "For loop range and step must be integers",
                context
            ))    

        var_name = node.var_name_token.value

        i = start_value.value
        
        condition = lambda: i < end_value.value if step_value.value >= 0 else i > end_value.value
        context.symbol_table.set(var_name, 'Int', Number(i))

        while condition():
            elements.append(rt_res.register(self.visit(node.body_node, context)))
            if rt_res.error: return rt_res
            i += step_value.value

            context.symbol_table.set(var_name, 'Int', Number(i))

        return rt_res.success(
            Void() if node.retr_void else List(elements).set_context(context).set_pos(node.pos_start, node.pos_end)
            )
    
    def visit_WhileNode(self, node, context):
        rt_res = RuntimeResult()
        elements = []
        while True:
            condition = rt_res.register(self.visit(node.condition_node, context))
            if rt_res.error: return rt_res

            if not condition.is_true(): break

            elements.append(rt_res.register(self.visit(node.body_node, context)))
            if rt_res.error: return rt_res

        return rt_res.success(Void() if node.retr_void else List(elements).set_context(context).set_pos(node.pos_start, node.pos_end))

    def visit_NumberNode(self, node, context):
        number = Number(node.token.value)
        number.set_context(context)
        number.set_pos(node.pos_start, node.pos_end)
        return RuntimeResult().success(number)
    
    def visit_StringNode(self, node, context):
        string = String(node.token.value)
        string.set_context(context)
        string.set_pos(node.pos_start, node.pos_end)
        return RuntimeResult().success(string)
    
    def visit_ListNode(self, node, context):
        rt_res = RuntimeResult()
        elements = []
        for element in node.elements: 
            elements.append(rt_res.register(self.visit(element, context)))
            if rt_res.error: return rt_res

        return rt_res.success(List(elements).set_context(context).set_pos(node.pos_start, node.pos_end))
    
    def visit_IndexNode(self, node, context):
        rt_res = RuntimeResult()

        parent = rt_res.register(self.visit(node.list_node, context))
        if rt_res.error: return rt_res

        if not isinstance(parent, (List, String)):
            return rt_res.failure(RuntimeError(
                node.pos_start, node.pos_end,
                "Can only index into a List",
                context
            ))

        
        start_idx = None if node.index_node is None else rt_res.register(self.visit(node.index_node, context))
        if rt_res.error: return rt_res

        if node.is_range:
            
            end = None if node.end is None else rt_res.register(self.visit(node.end, context))
            if rt_res.error: return rt_res

            
            step = None if node.step is None else rt_res.register(self.visit(node.step, context))
            if rt_res.error: return rt_res

            index = Range(start_idx, end, step, node.pos_start, node.pos_end)
        else:
            index = start_idx

        result, error = parent.access(index)
        if error:
            return rt_res.failure(error)
        
        # return rt_res.success(result.set_pos(node.pos_start, node.pos_end).set_context(context))
        return rt_res.success(result)

    def visit_IFNode(self, node, context):
        rt_res = RuntimeResult()
        
        for condition, expr, retr_void in node.cases : 
            condition_value  = rt_res.register(self.visit(condition, context))
            if rt_res.error: return rt_res

            if condition_value.is_true():
                expr_value = rt_res.register(self.visit(expr, context))
                if rt_res.error: return rt_res
                return rt_res.success(Void() if retr_void else expr_value) 
        
        if node.else_case:
            expr, retr_void = node.else_case
            else_value  = rt_res.register(self.visit(expr, context))
            if rt_res.error: return rt_res
            return rt_res.success(Void() if retr_void else else_value) 

        return rt_res.success(Void())
    
    def visit_BinOpNode(self, node, context):
        rt_res = RuntimeResult()

        left = rt_res.register(self.visit(node.left_node, context))
        if rt_res.error: return rt_res
        right = rt_res.register(self.visit(node.right_node, context))
        if rt_res.error: return rt_res


        # Type safety.

        if isinstance(left, Bool):
            left = self.numerize(left)
        if isinstance(right, Bool):
            right = self.numerize(right)

        if isinstance(left, List):
            if node.opr_token.type == TT_LSHIFT:  # << operator
                result, error = left.append(right)
            
            elif node.opr_token.type == TT_RSHIFT:  # >> operator
                result, error = left.remove(right)

            elif node.opr_token.type == TT_PLUS:  # + operator
                result, error = left.add(right)
            else:
                return rt_res.failure(RuntimeError(
                    node.pos_start, node.pos_end,
                    f"Operator '{node.opr_token.type}' not supported for List",
                    context
                ))
            
        if node.opr_token.type == TT_PLUS:
            result, error = left.add(right)

        elif node.opr_token.type == TT_MINUS:
            result, error = left.subtract(right)

        elif node.opr_token.type == TT_MUL:
            result, error = left.multiply(right)

        elif node.opr_token.type == TT_DIV:
            result, error = left.divide(right)

        elif node.opr_token.type == TT_POW:
            result, error = left.raise_to(right)

        elif node.opr_token.type == TT_EE:
            result, error = left.get_comparison_eq(right)
            if result: result = Bool(result)

        elif node.opr_token.type == TT_NE:
            result, error = left.get_comparison_ne(right)
            if result: result = Bool(result)

        elif node.opr_token.type == TT_LT:
            result, error = left.get_comparison_lt(right)
            if result: result = Bool(result)

        elif node.opr_token.type == TT_GT:
            result, error = left.get_comparison_gt(right)
            if result: result = Bool(result)        

        elif node.opr_token.type == TT_LTE:
            result, error = left.get_comparison_lte(right)
            if result: result = Bool(result)

        elif node.opr_token.type == TT_GTE:
            result, error = left.get_comparison_gte(right)
            if result: result = Bool(result)

        elif node.opr_token.matches(TT_KWORD, 'and'):
            result, error = left.and_(right)
            # if result: result = Bool(result)

        elif node.opr_token.matches(TT_KWORD, 'or'):
            result, error = left.or_(right)
                # if result: result = Bool(result) 
        
        if error:
            return rt_res.failure(error)
        else:
            succ = rt_res.success(result.set_pos(node.pos_start, node.pos_end))
            return succ

    def visit_UnaryOpNode(self, node, context):
        
        rt_res = RuntimeResult()
        value = rt_res.register(self.visit(node.node, context))
        if rt_res.error: return rt_res

        error = None
        
        if node.opr_token.type == TT_MINUS:
            minus_1 = Number(-1)
            result, error = value.multiply(minus_1)
        
        elif node.opr_token.matches(TT_KWORD, 'not'): # Type safety
            if not isinstance(value, Bool):
                return rt_res.failure(RuntimeError(node.pos_start, node.pos_end, f"Operator 'not' expects a Bool operand, got {type(value).__name__}", context))
            
            result, error = value.not_()


        if error: return rt_res.failure(error)
        else:
            return rt_res.success(result.set_pos(node.pos_start, node.pos_end))
        
    def visit_VarAccessNode(self, node, context):
        res = RuntimeResult()
        var_name = node.var_name.value
        
        var_type, var_value = context.symbol_table.get(var_name)

        if not var_value and not var_type:
            return res.failure(RuntimeError(node.pos_start, node.pos_end, f"'{var_name}' is not defined", context))
        elif not var_value and var_type != None:
            return res.failure(RuntimeError(node.pos_start, node.pos_end, f"'{var_name}' is uninitialized", context))

        var_value = var_value.copy().set_pos(node.pos_start, node.pos_end).set_context(context)

        return res.success(var_value)

    def visit_VarAssignNode(self, node, context):
        res = RuntimeResult()
        var_name = node.var_name.value
        var_type_token = node.var_type_node  
        var_type = var_type_token.value if var_type_token else None  

        existing_type, existing_value = context.symbol_table.get(var_name)
        
        if node.value_node :
            if isinstance(node.value_node, VarAssignNode) and node.value_node.value_node is None:
                # New check to make sure an uninitialized var is not being assigned to another uninitialized var
                return res.failure(RuntimeError(node.pos_start, node.pos_end, "Cannot assign an uninitialized declaration", context))
            
            var_value =  res.register(self.visit(node.value_node, context))
            if res.error: return res
        else:
            var_value = None    

        new_type = None
        if var_value is not None:
            if isinstance(var_value, Number):
                new_type = 'Int' if isinstance(var_value.value, int) else 'Float'
            elif isinstance(var_value, String):
                new_type = 'String'
            elif isinstance(var_value, Bool):
                new_type = 'Bool'
            elif isinstance(var_value, List):
                new_type = 'List'
            elif isinstance(var_value, Void):
                new_type = 'Void'
            else:
                return res.failure(RuntimeError(node.pos_start, node.pos_end, f"Unsupported type for assignment: {type(var_value).__name__}", context))

        if existing_type:
            if var_type_token:
                # Redeclaration; allow shadowing, check new type if value provided
                if var_value is not None and new_type != var_type:
                    return res.failure(RuntimeError(node.pos_start, node.pos_end, f"Cannot assign {new_type} to variable '{var_name}' declared as {var_type}", context))
            else:
                # Assignment; enforce existing type
                if var_value is not None and new_type != existing_type:
                    return res.failure(RuntimeError(node.pos_start, node.pos_end, f"Cannot assign {new_type} to variable '{var_name}' of type {existing_type}", context))
                var_type = existing_type
        else:
            # New variable
            if var_type is not None and var_value is not None:
                if new_type != var_type:
                    return res.failure(RuntimeError(node.pos_start, node.pos_end, f"Cannot assign {new_type} to variable '{var_name}' declared as {var_type}", context))
                
            elif var_type is None and var_value is not None:
                var_type = new_type  # Type inference

            elif var_type is not None and var_value is None:
                pass
            else:
                return res.failure(RuntimeError(node.pos_start, node.pos_end, f"Cannot declare '{var_name}' without a type or value", context))

        context.symbol_table.set(var_name, var_type, var_value)
        return res.success(var_value)  


    def visit_FunDefNode(self, node, context):
        rt_res = RuntimeResult()

        fun_name =  node.var_name_token.value if node.var_name_token else None
        body_node = node.body_node
        arg_names = [arg_name.value for arg_name in node.arg_name_tokens]

        func_value = Function(fun_name, body_node, arg_names, node.retr_void).set_context(context).set_pos(node.pos_start, node.pos_end)
        
        if node.var_name_token:
            context.symbol_table.set(fun_name, 'Function' ,func_value)

        return rt_res.success(func_value)
    
    def visit_CallNode(self, node, context):
        rt_res = RuntimeResult()
        args = []

        to_call = rt_res.register(self.visit(node.node_to_call, context))
        if rt_res.error: return rt_res

        to_call = to_call.copy().set_pos(node.pos_start, node.pos_end)
        
        for arg_node in node.arg_nodes:
            args.append(rt_res.register(self.visit(arg_node, context)))
            if rt_res.error: return rt_res

        return_value = rt_res.register(to_call.execute(args))
        if rt_res.error: return rt_res

        return_value = return_value.copy().set_pos(node.pos_start, node.pos_end).set_context(context)
        return rt_res.success(return_value)


global_symbol_table = SymbolTable()
global_symbol_table.set('Void', 'Void', Void())
global_symbol_table.set('True', 'Bool',Bool(1))
global_symbol_table.set('False','Bool', Bool(0))

global_symbol_table.set('print', 'Function', BuiltInFunction.print)
global_symbol_table.set('input', 'Function', BuiltInFunction.input)
global_symbol_table.set('clear', 'Function', BuiltInFunction.clear)
global_symbol_table.set('run', 'Function', BuiltInFunction.run)
global_symbol_table.set('exit', 'Function', BuiltInFunction.exit)

# global_symbol_table.set('Aditya','String', String('He is the best!'))

def future(): # time travel?
    return global_symbol_table.symbols

# Run
def run(filename, text):
    # Lexer
    lexer = Lexer(filename, text)
    tokens, error = lexer.tokenize()
    
    if error: return None, error, None

    # Generate AST (Abstract Syntax Tree)
    parser = Parser(tokens, future())
    ast = parser.parse()
    if ast.error: return None, ast.error
    
    # Run Program
    interpreter = Interpreter()
    context = Context('<program>')
    context.symbol_table = global_symbol_table
    
    result =  interpreter.visit(ast.node, context)
    res = result.value if result.value is not None else None
    
    return res, result.error
