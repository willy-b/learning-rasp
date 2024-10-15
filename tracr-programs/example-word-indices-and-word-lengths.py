# note Tracr is not for performance but for improving the theoretical
# understanding of Transformers by allowing explicit programming of operations
# that the Transformers neural network architecture (the T in GPT) can learn.
from tracr.rasp import rasp
from tracr.compiler import compiling
from tracr.compiler.lib import make_frac_prevs
import math

# In Weiss' RASP one can use `selector_width(select(tokens, " ", ==) and select(indices, indices, <=))` to count spaces up to a point;
# or even if one does do aggregate to get fraction of spaces up to that point, one can multiply by the index to get number of spaces (not supported in Tracr).
# It seems `selector_and` of two selectors that don't have the same queries and keys cannot be used with `selector_width` in Tracr at this time.
# Also pointwise multiplication between two sequences is not supported in Tracr at this time.
# So below we use Tracr built-in aggregate to get frac_prevs and since cannot multiply by indices,
# we take the logarithm of both sequences, add them, and then exponentiate to accomplish the multiplication.
def count_occurrences_up_to(sop):
  # sop could be 'rasp.tokens == " "' for example
  frac_prevs_sop = make_frac_prevs(sop)
  # 10 is chosen as some number guaranteed greater than math.log(MAX_SEQUENCE_LENGTH + 1)
  log_frac_prevs_sop = rasp.numerical(rasp.Map(lambda x: math.log(x) if x > 0 else 10, rasp.numerical(frac_prevs_sop)))
  log_indices = rasp.numerical(rasp.Map(lambda x: math.log(x + 1), rasp.indices))
  log_sum_frac_prevs_sop_and_indices = rasp.numerical(rasp.LinearSequenceMap(log_frac_prevs_sop, log_indices, 1, 1))
  num_occurrences_up_to = rasp.Map(lambda x: round(math.exp(x)) if x < 10 else 0, log_sum_frac_prevs_sop_and_indices)
  return num_occurrences_up_to

# Note in Weiss' RASP these operations are easier than in Tracr's subset supported operations
# word_indices = (1+selector_width(select(tokens, " ", ==) and select(indices, indices, <=)))*(0 if indicator(tokens == " ") else 1);
word_indices = rasp.SequenceMap(lambda fst, snd: 0 if (snd == ' ' or snd == '.') else fst, 
                                count_occurrences_up_to(rasp.tokens == " ") + 1, 
                                rasp.tokens
                                )
# Weiss' RASP:
# word_lengths = selector_width(select(word_indices, word_indices, ==))*(0 if indicator(word_indices <= 0) else 1);
word_lengths = rasp.SequenceMap(lambda fst, snd: 0 if (snd == ' ' or snd == '.') else fst , 
                                rasp.SelectorWidth(rasp.Select(word_indices, word_indices, rasp.Comparison.EQ)), 
                                rasp.tokens
                                )
print("Tracr is compiling RASP language to a Transformer neural network model for 'word_indices_model', please wait...")
word_indices_model = compiling.compile_rasp_to_model(
  word_indices,
  vocab=set(['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','0','1','2','3','4','5','6','7','8','9',' ',',','.']),
  max_seq_len=30,
  compiler_bos='BOS'
)
example_sentence_input = ['BOS', 't', 'h', 'e', ' ', 'b', 'o', 'y', ' ', 'p', 'a', 'i', 'n', 't', 'e', 'd', ' ', 't', 'h', 'e', ' ', 'g', 'i', 'r', 'l', '.']
print(f"Input: {example_sentence_input}")
word_indices_output = word_indices_model.apply(example_sentence_input).decoded
print(f"Marking word indices with Transformer hand-coded in Tracr RASP:\n{word_indices_output}")
# [BOS,t,h,e, ,b,o,y, ,p,a,i,n,t,e,d, ,t,h,e, ,g,i,r,l,.]
# [BOS,1,1,1,0,2,2,2,0,3,3,3,3,3,3,3,0,4,4,4,0,5,5,5,5,0]

print("Tracr is compiling RASP language to a Transformer neural network model for 'word_lengths_model', please wait...")
word_lengths_model = compiling.compile_rasp_to_model(
  word_lengths,
  vocab=set(['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','0','1','2','3','4','5','6','7','8','9',' ',',','.']),
  max_seq_len=30,
  compiler_bos='BOS'
)
word_lengths_output = word_lengths_model.apply(example_sentence_input).decoded
print(f"Marking word lengths with Transformer hand-coded in Tracr RASP:\n{word_lengths_output}")
# [BOS,t,h,e, ,b,o,y, ,p,a,i,n,t,e,d, ,t,h,e, ,g,i,r,l,.]
# [BOS,3,3,3,0,3,3,3,0,7,7,7,7,7,7,7,0,3,3,3,0,4,4,4,4,0]
