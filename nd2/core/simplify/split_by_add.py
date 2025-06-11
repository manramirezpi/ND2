from ..base_visitor import Visitor
from ..symbols import Symbol, Number, Add, Sub, Mul, Div, Aggr, Rgga, Sour, Targ

class SplitByAdd(Visitor):
    def __call__(self,
                node: Symbol,
                split_by_sub: bool = False,
                expand_mul: bool = False,
                expand_div: bool = False,
                expand_aggr: bool = False,
                expand_rgga: bool = False,
                expand_sour: bool = False,
                expand_targ: bool = False,
                remove_coefficients: bool = False,
                merge_bias: bool = False) -> list[Symbol]:
        """Split the node by addition, returning a list of symbols.
        Args:
        - node: Symbol, the node to split
        - split_by_sub: bool, whether to split by Sub nodes
        - expand_mul: bool, whether to expand Mul nodes
        - expand_div: bool, whether to expand Div nodes
        - expand_aggr: bool, whether to expand Aggr nodes
        - expand_rgga: bool, whether to expand Rgga nodes
        - expand_sour: bool, whether to expand Sour nodes
        - expand_targ: bool, whether to expand Targ nodes
        - remove_coefficients: bool, whether to remove coefficients from the symbols
        - merge_bias: bool, whether to merge bias terms
        """
        return super().__call__(node,
                                split_by_sub=split_by_sub,
                                expand_mul=expand_mul,
                                expand_div=expand_div,
                                expand_aggr=expand_aggr,
                                expand_rgga=expand_rgga,
                                expand_sour=expand_sour,
                                expand_targ=expand_targ,
                                remove_coefficients=remove_coefficients,
                                merge_bias=merge_bias)
    
    def generic_visit(self, node: Symbol, *args, **kwargs) -> list[Symbol]:
        return [node]
    
    def visit_Add(self, node: Add, *args, **kwargs) -> list[Symbol]:
        x1, x2 = node.operands
        result = self(x1, *args, **kwargs) + self(x2, *args, **kwargs)
        if kwargs.get('merge_bias'):
            result = self.merge_bias(result, *args, **kwargs)
        return result
    
    def visit_Sub(self, node: Sub, *args, **kwargs) -> list[Symbol]:
        if not kwargs.get('split_by_sub'): return [self]
        x1, x2 = node.operands
        result = self(x1, *args, **kwargs) + self(x2, *args, **kwargs)
        if kwargs.get('merge_bias'):
            result = self.merge_bias(result, *args, **kwargs)
        return result

    def visit_Mul(self, node: Mul, *args, **kwargs) -> list[Symbol]:
        if not kwargs.get('expand_mul'): return [node]
        x1, x2 = node.operands
        result1 = self(x1, *args, **kwargs)
        result2 = self(x2, *args, **kwargs)
        result = []
        for item in result1:
            for jtem in result2:
                if not kwargs.get('remove_coefficients'):
                    result.append(item * jtem)
                elif isinstance(item, Number): 
                    result.append(jtem)
                elif isinstance(jtem, Number): 
                    result.append(item)
                else: 
                    result.append(item * jtem)
        if kwargs.get('merge_bias'):
            result = self.merge_bias(result, *args, **kwargs)
        return result

    # def visit_Div(self, node: Div, *args, **kwargs) -> list[Symbol]:
    #     if not kwargs.get('expand_div'): return [node]
    #     x1, x2 = node.operands
    #     result1 = self(x1, *args, **kwargs)
    #     result2 = [x2]
    #     result = []
    #     for item in result1:
    #         for jtem in result2:
    #             if not kwargs.get('remove_coefficients'):
    #                 result.append(item / jtem)
    #             elif isinstance(item, Number): 
    #                 result.append(jtem)
    #             elif isinstance(jtem, Number): 
    #                 result.append(item)
    #             else: 
    #                 result.append(item / jtem)
    #     if kwargs.get('merge_bias'):
    #         result = self.merge_bias(result, *args, **kwargs)
    #     return result

    def visit_Sour(self, node: Sour, *args, **kwargs) -> list[Symbol]:
        if not kwargs.get('expand_sour'): return [node]
        result = self(node.operands[0], *args, **kwargs)
        if kwargs.get('merge_bias'):
            result = self.merge_bias(result, *args, **kwargs)
        for idx, item in enumerate(result):
            if isinstance(item, Number): result[idx] = item # 因为 Sour(C) 和 C 数学等价
            else: result[idx] = Sour(item)
        return result

    def visit_Targ(self, node: Targ, *args, **kwargs) -> list[Symbol]:
        if not kwargs.get('expand_targ'): return [node]
        result = self(node.operands[0], *args, **kwargs)
        if kwargs.get('merge_bias'):
            result = self.merge_bias(result, *args, **kwargs)
        for idx, item in enumerate(result):
            if isinstance(item, Number): result[idx] = item # 因为 Targ(C) 和 C 数学等价
            else: result[idx] = Targ(item)
        return result
    
    def visit_Aggr(self, node: Aggr, *args, **kwargs) -> list[Symbol]:
        if not kwargs.get('expand_aggr'): return [node]
        result = self(node.operands[0], *args, **kwargs)
        if kwargs.get('merge_bias'):
            result = self.merge_bias(result, *args, **kwargs)
        for idx, item in enumerate(result):
            result[idx] = Aggr(item) # Aggr(C) 和 C 数学不等价
        return result

    def visit_Rgga(self, node: Rgga, *args, **kwargs) -> list[Symbol]:
        if not kwargs.get('expand_rgga'): return [node]
        result = self(node.operands[0], *args, **kwargs)
        if kwargs.get('merge_bias'):
            result = self.merge_bias(result, *args, **kwargs)
        for idx, item in enumerate(result):
            result[idx] = Rgga(item) # Rgga(C) 和 C 数学不等价
        return result

    def merge_bias(self, items: list[Symbol], *args, **kwargs) -> list[Symbol]:
        """Merge bias terms in the node."""
        is_bias = [isinstance(item, Number) for item in items]
        if not any(is_bias): return items
        bias = Number(sum(items[idx].value for idx, flag in enumerate(is_bias) if flag))
        if kwargs.get('remove_coefficients'): bias = Number(1.0)
        items_merge_bias = [bias] + [items[idx] for idx, flag in enumerate(is_bias) if not flag]
        return items_merge_bias