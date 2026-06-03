from typing import List, Dict

class Solution:
    def tokenize_numbers(self, numbers: List[int], vocab: Dict[str, int]) -> List[List[str]]:
        # Tokenize each number using greedy left-to-right longest match.
        # Return a list of token lists showing how each number gets split.
        ans=[]
        for x in numbers:
            s=str(x)
            tokens=self._greedy_tokenize(s, vocab)
            ans.append(tokens)
        return ans
        pass

    def count_tokens(self, text: str, vocab: Dict[str, int]) -> int:
        # Count how many tokens the text uses with greedy tokenization.
        # Use greedy left-to-right longest match.
        tokens=self._greedy_tokenize(text, vocab)
        return len(tokens)
        pass

    def fertility_score(self, text: str, vocab: Dict[str, int]) -> float:
        # Compute tokens-per-word ratio (fertility).
        # Higher = more expensive and less efficient.
        # Round to 4 decimal places.
        tokens=self._greedy_tokenize(text, vocab)
        words=text.split()
        return round(len(tokens)/len(words), 4)
        pass
    
    def _greedy_tokenize(self, text: str, vocab: Dict[str, int]) -> List[List[str]]:
        tokens=[]
        i=0
        while i< len(text):
            best=""
            for j in range(len(text)-i, 0, -1):
                substr=text[i:i+j]
                if substr in vocab:
                    best=substr
                    break
            if best == "":
                tokens.append(text[i])
                i+=1
            else:
                tokens.append(best)
                i+=(len(best))
        return tokens
