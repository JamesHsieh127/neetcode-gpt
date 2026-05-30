from typing import Dict, List, Tuple

class Solution:
    def build_vocab(self, text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
        # Return (stoi, itos) where:
        # - stoi maps each unique character to a unique integer (sorted alphabetically)
        # - itos is the reverse mapping (integer to character)
        chars=sorted(set(text))
        stoi={}
        itos={}
        for i in range(0, len(chars), 1):
            stoi[chars[i]]=i
            itos[i]=chars[i]
        return stoi, itos
        pass

    def encode(self, text: str, stoi: Dict[str, int]) -> List[int]:
        # Convert a string to a list of integers using stoi mapping
        ans=[]
        for i in range(0, len(text), 1):
            ans.append(stoi[text[i]])
        return ans
        pass

    def decode(self, ids: List[int], itos: Dict[int, str]) -> str:
        # Convert a list of integers back to a string using itos mapping
        ans=[]
        for i in range(0, len(ids), 1):
            ans.append(itos[ids[i]])
        return "".join(ans)
        pass
