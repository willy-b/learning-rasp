# Show PARITY for fixed length input can be coded this way
# Interesting: It is very easy to program the PARITY task in Tracr compatible RASP,
# where RASP intends to model the computational capabilities of Transformer encoders,
# but Transformer encoders are not generally regarded as being able to actually
# learn to solve PARITY
# (unlike a feedforward neural network in general as
# Chiang and Cholak 2022 ( https://arxiv.org/abs/2202.12172 ) note that
# Rumelhart 1986 showed).
# Definition of PARITY per Strobl et al 2024
# ( https://arxiv.org/abs/2311.00208 ):
# "The classic examples of languages not in AC0 are PARITY and MAJORITY.
# The language PARITY ⊆ {0, 1}∗ contains all bit strings
# containing an odd number of 1’s, and MAJORITY ⊆ {0, 1}∗ consists of
# all bit strings in which more than half of the bits are 1’s."
# With intermediate decoder steps Transformers are Turing complete and
# can definitely solve Parity (though it is even then hard to train,
# see Zhou et al 2024 ( https://arxiv.org/abs/2310.16028 )
# who modify RASP to get learnable RASP (RASP-L) and
# apply RASP-L to autoregressive decoder with scratchpad),
# but RASP per Weiss 2021 ( https://arxiv.org/abs/2106.06981 )
# originally models an encoder WITHOUT a decoder
# loop/scratchpad/chain of thought and RASP solves PARITY easily.
# Strobl et al 2024 note that "Average-hard and softmax attention"
# as opposed to hard attention enable counting
# "which can be used to solve problems like MAJORITY that are outside AC0"
# (like PARITY as well), but
# Chiang and Cholak (2022) ( https://arxiv.org/abs/2202.12172 )  show that
# while they theoretically can implement parity with a Transformer encoder
# they could NOT train a Transformer to learn it,
# and Zhou et al 2024 also excluded RASP operations when defining their
# RASP-L (Learnable RASP) such that parity was not solvable
# without modification even with a decoder loop.
# Deletang et al 2023 ( https://arxiv.org/abs/2207.02098 )
# also find that "transformers learn MAJORITY, but not PARITY."

from tracr.rasp import rasp
length_constant = 10
# can't do the operation that works in Weiss RASP, need to adapt to a Tracr compatible way
# In Weiss' RASP: `parity = selector_width(select(tokens, "1", ==)) % 2;`
# but cannot have "1" or 1 as 2nd argument in Tracr select;
# also cannot aggregate to get fraction of ones and multiply by length does not work due to numerical stability in Tracr (not in RASP interpreter from Weiss) (multiplying 0.30 by 10 for example gives 3.33 in Tracr)
# turns out the Tracr preferred way is to define a lambda to use as the comparator and ignore the 2nd argument:
ones_selector = rasp.Select(rasp.tokens, rasp.tokens, lambda k, q: k == 1)
count_ones = rasp.SelectorWidth(ones_selector)

parity = rasp.Map(lambda x: x % 2, count_ones)

from tracr.compiler import compiling

bos = "BOS"
pad = "PAD"

model_count_ones = compiling.compile_rasp_to_model(
    count_ones,
    vocab={0, 1},
    max_seq_len=int(length_constant),
    compiler_bos=bos,
    compiler_pad=pad,
)

model_parity = compiling.compile_rasp_to_model(
    parity,
    vocab={0, 1},
    max_seq_len=int(length_constant),
    compiler_bos=bos,
    compiler_pad=pad
)

# Show PARITY for fixed length input can be coded this way
# should handle sequences up to length "length_constant" - 1

# Some examples
sequence = [1, 0,0,0,1]
output = model_parity.apply([bos] + sequence).decoded
print(f"Transformer model computes PARITY of {sequence} as {output} (take any position after first in output sequence as answer)")
# ['BOS', 0, 0, 0, 0, 0]

sequence = [1, 0,1,0,1]
output = model_parity.apply([bos] + sequence).decoded
print(f"Transformer model computes PARITY of {sequence} as {output} (take any position after first in output sequence as answer)")
# ['BOS', 1, 1, 1, 1, 1]

sequence = [1, 1,0,1,1]
output = model_parity.apply([bos] + sequence).decoded
print(f"Transformer model computes PARITY of {sequence} as {output} (take any position after first in output sequence as answer)")
# ['BOS', 0, 0, 0, 0, 0]

sequence = [0, 1,1,1,0]
output = model_parity.apply([bos] + sequence).decoded
print(f"Transformer model computes PARITY of {sequence} as {output} (take any position after first in output sequence as answer)")
# ['BOS', 1, 1, 1, 1, 1]

sequence = [1, 0,0,0,0]
output = model_parity.apply([bos] + sequence).decoded
print(f"Transformer model computes PARITY of {sequence} as {output} (take any position after first in output sequence as answer)")
# ['BOS', 1, 1, 1, 1, 1]

sequence = [0, 0,0,0,0]
output = model_parity.apply([bos] + sequence).decoded
print(f"Transformer model computes PARITY of {sequence} as {output} (take any position after first in output sequence as answer)")
# ['BOS', 0, 0, 0, 0, 0]

sequence = []
for idx in range(int(length_constant - 1)):
    sequence.append(1)
    inp = [bos] + sequence
    output = model_parity.apply(inp).decoded
    output_count_ones = model_count_ones.apply(inp).decoded
    print(f"Transformer model computes count of ones of {sequence} as {output_count_ones}")
    print(f"Transformer model computes PARITY of {sequence} as {output} (take any position after first in output sequence as answer)")
