from typing import Tuple
import IPython

from reward_machines.rm_common import RewardMachine
from .regex_ast import RENode, NodeCreator
from .regex_compiler import CompileStateNFA, CompileStateDFA, to_dfa
from . import regex_parser
from . import regex_lexer


def lex(src: str) -> list[regex_parser.Token]:
    return list(regex_lexer.lex(src))


def parse(src: str) -> RENode:
    return regex_parser.parse(src)


def get_nfa(src: str) -> Tuple[CompileStateNFA, NodeCreator]:
    ast: RENode = parse(src)
    appears: frozenset[str] = ast.appears()
    node_creator: NodeCreator = NodeCreator(appears)
    return ast.compile(node_creator).relabel_states(), node_creator


def get_dfa(src: str) -> Tuple[CompileStateDFA, NodeCreator]:
    nfa, node_creator = get_nfa(src)
    return to_dfa(nfa, node_creator).relabel_states(), node_creator


def compile(src: str) -> RewardMachine:
    dfa, node_creator = get_dfa(src)
    return dfa_to_rm(dfa, node_creator.appears)
