import os
import pty
import subprocess
import pandas as pd
import numpy as np
import argparse

argparser = argparse.ArgumentParser()
# this only produces a score on the training set of ReCOGS by default
argparser.add_argument("--num_train_examples_to_check", type=int, default=5)
argparser.add_argument("--use_dev_split", action="store_true")
args = argparser.parse_args()

base_path = os.path.abspath(".")
# Load dependency if not available.
# we use the Restricted Access Sequence Processing interpreter from "Thinking Like Transformers" Weiss et al 2021 ( https://arxiv.org/abs/2106.06981 )
# This RASP is an academic language which can be theoretically compiled to Transformer neural network weights, useful for thinking about Transformer capabilities.
print("Note the RASP dependency requires python3.10-venv and its own dependencies; if there are errors please check the RASP dependencies' instructions (and run `apt install python3.10-venv` or equivalent for your operating system)")
if not os.path.exists(base_path + "/RASP"):
  subprocess.run("git clone https://github.com/tech-srl/RASP.git", shell=True, executable='/bin/bash')
  os.chdir(base_path + "/RASP")
  subprocess.run(base_path + "/RASP/setup.sh", shell=True, executable="/bin/bash")
  os.chdir(base_path)

with open(base_path + "/RASP/rasp2.sh", "w") as f:
  f.write("""#!/bin/bash
source raspenv/bin/activate
python3 -m RASP_support
deactivate
""")
os.chmod(base_path + "/RASP/rasp2.sh", 750)

os.chdir(base_path + "/RASP")
main, secondary = pty.openpty()
proc = subprocess.Popen(base_path + "/RASP/rasp2.sh", shell=True, executable='/bin/bash', stdin=subprocess.PIPE, stdout=secondary)
print(proc)
stdin_handle = proc.stdin
stdout_handle = os.fdopen(main)

with open(base_path + "/" + "word-level-pos-tokens-recogs-style-decoder-loop.rasp", "r") as f:
  rasp_setup_lines = f.readlines()
input_lines = [bytes(line, 'utf8') for line in rasp_setup_lines]

stdin_handle.writelines(input_lines)
stdin_handle.flush()

# discard all output from running the setup part of the script
input_lines = ["autoregressive_output = output;\n"]
input_lines = [bytes(line, 'utf8') for line in input_lines]
stdin_handle.writelines(input_lines)
stdin_handle.flush()
outputline = stdout_handle.readline()
while outputline.find("Example: autoregressive_output") < 0:
  outputline = stdout_handle.readline()

def process_example(example, suppress_output=True, debug_mode=False):
  current_example = example.lower().split(" ")
  nextcharacter = " "
  while nextcharacter != "":
    next_input = f"set example {current_example}\nautoregressive_output;\n".replace("'", '"')
    input_lines = [next_input]
    input_lines = [bytes(line, 'utf8') for line in input_lines]
    stdin_handle.writelines(input_lines)
    stdin_handle.flush()
    outputline = stdout_handle.readline()
    while outputline.find("Example: autoregressive_output") < 0:
      if debug_mode:
        print(f"skipping: {outputline}")
      outputline = stdout_handle.readline()
    if not suppress_output:
      print(outputline)
    nextcharacter = outputline.split("=")[1].split("m[")[1].split("]")[0]
    current_example.append(nextcharacter)
  translation = " ".join(current_example).split("|")
  if not suppress_output:
    print(f"{translation[0]}\n{translation[1]}")
  return translation[1]

print("""

Note, it is simpler and more performant to just train the Transformer on examples!
This is an academic exercise, writing a neural network compatible program
by hand in the Restricted Access Sequence Processing (compilable to Transformer)
language (Weiss et al 2021, https://arxiv.org/abs/2106.06981 ) to work towards
eventually proving a Transformer can perform a particular type of solution
(we will be building a grammar based and compositional solution
we can prove things about).

""")

print("Run one example (two semantically identical but syntactically different forms) before loading the dataset:")

process_example("a boy painted the girl", False)

process_example("the girl was painted by a boy", False)
recogs_datafile = None

score_on_train_sample = not args.use_dev_split

print(f"Fetching script for ReCOGS Semantic Exact Match scoring which is more flexible than exact string match (ignores irrelevant formatting differences and ordering) from Wu et al 2023's repo...")
if not os.path.exists(base_path + "/compgen.py"):
  os.chdir(base_path)
  subprocess.run("wget https://raw.githubusercontent.com/frankaging/ReCOGS/1b6eca8ff4dca5fd2fb284a7d470998af5083beb/utils/compgen.py", shell=True, executable='/bin/bash')
from compgen import recogs_exact_match
def get_semantic_exact_match_score(lfs_predicted, lfs_actual):
  # for semantic exact match we must NOT lowercase the "AND" as it searches for the pattern in a case-sensitive way
  semantic_exact_matches = np.array([1.0 if recogs_exact_match(lfs_predicted[idx].strip().lower().replace(" and ", " AND "),lfs_actual[idx].strip().lower().replace(" and ", " AND ")) else 0.0 for idx in range(len(lfs_predicted))])
  mean_semantic_exact_match_score = semantic_exact_matches.mean()
  num_right_semantic_exact_match = semantic_exact_matches.sum()
  return mean_semantic_exact_match_score, num_right_semantic_exact_match, semantic_exact_matches

if score_on_train_sample:
  print("Now load official Wu et al 2023 ReCOGS training examples\n(sample from https://raw.githubusercontent.com/frankaging/ReCOGS/refs/heads/main/recogs_positional_index/train.tsv , associated with https://arxiv.org/abs/2303.13716 )")
  if not os.path.exists("train_in_distribution_no_sprinkles_or_cp.tsv"):
    # one of official author's dataset for ReCOGS paper
    subprocess.run("wget https://raw.githubusercontent.com/frankaging/ReCOGS/refs/heads/main/recogs_positional_index/train.tsv", shell=True)
    subprocess.run("echo 'COGS Sentence	ReCOGS Logical Form	Distribution' > train_in_distribution_no_sprinkles_or_cp.tsv", shell=True)
    subprocess.run("cat train.tsv | grep 'in_distribution' | grep -v 'sprinkle' | grep -v 'that' >> train_in_distribution_no_sprinkles_or_cp.tsv", shell=True)
  recogs_datafile = pd.read_csv("train_in_distribution_no_sprinkles_or_cp.tsv", delimiter="	")
else:
  print("Using dev split, the `num_train_examples_to_check` argument will be ignored")
  print("Now load official Wu et al 2023 ReCOGS dev split (excluding complement phrases as not yet supported)\n(https://raw.githubusercontent.com/frankaging/ReCOGS/refs/heads/main/recogs_positional_index/dev.tsv , associated with https://arxiv.org/abs/2303.13716 )")
  if not os.path.exists("dev_no_cp.tsv"):
    # one of official author's dataset for ReCOGS paper
    subprocess.run("wget https://raw.githubusercontent.com/frankaging/ReCOGS/refs/heads/main/recogs_positional_index/dev.tsv", shell=True)
    subprocess.run("echo 'COGS Sentence	ReCOGS Logical Form	Distribution' > dev_no_cp.tsv", shell=True)
    subprocess.run("cat dev.tsv | grep -v 'that' >> dev_no_cp.tsv", shell=True)
  recogs_datafile = pd.read_csv("dev_no_cp.tsv", delimiter="	")

if score_on_train_sample:
  sentences = recogs_datafile["COGS Sentence"][:args.num_train_examples_to_check]
  lfs_true = recogs_datafile["ReCOGS Logical Form"][:args.num_train_examples_to_check]
else:
  sentences = recogs_datafile["COGS Sentence"]
  lfs_true = recogs_datafile["ReCOGS Logical Form"]

sentences = [sentence.replace(" .", "").replace(".", "") for sentence in sentences]
lfs_computed = []
for idx in range(len(sentences)):
  lfs_computed.append(process_example(sentences[idx], False))
  if idx % 10 == 0:
    # update CSV (these are small so can rewrite each time)
    output_df = pd.DataFrame([{"Input Sentence": sentences[jdx], "Logical Form Predicted": lfs_computed[jdx]} for jdx in range(idx+1)], columns=["Input Sentence", "Logical Form Predicted"])
    output_df.to_csv("lf_output.tsv", index=False, sep="	")
    if len(lfs_computed)>=10:
      mean_semantic_exact_match_score, num_right_semantic_exact_match, _ = get_semantic_exact_match_score(lfs_computed, lfs_true[:len(lfs_computed)])
      print(f"Semantic match score on first {len(lfs_computed)} of ReCOGS_pos train:\n(omitting CP and sprinkles or preposing as not supported in the grammar of this Restricted Access Sequence Processing Transformer equivalent program yet)\n{mean_semantic_exact_match_score*100}% or {num_right_semantic_exact_match} out of {len(lfs_computed)}")

output_df = pd.DataFrame([{"Input Sentence": sentences[jdx], "Logical Form Predicted": lfs_computed[jdx]} for jdx in range(len(lfs_computed))], columns=["Input Sentence", "Logical Form Predicted"])
output_df.to_csv("lf_output.tsv", index=False, sep="	")

# note not finished with the grammar yet and it will, when scored by Exact Match, systematically miss the v_inf_taking_to_v_inf (e.g. "the scientist wanted to read") due to misordered output vs reference
# in ReCOGS "semantic exact match" they would still be correct as it ignores semantically meaningless reorderings but I am scoring with exact match here for the time being

exact_matches = [1.0 if lfs_computed[idx].strip().lower() == lfs_true[idx].strip().lower() else 0.0 for idx in range(len(lfs_computed))]
mean_em_score = np.array(exact_matches).mean()
num_right = np.array(exact_matches).sum()

if score_on_train_sample:
  print(f"Exact Match score on first {len(sentences)} of ReCOGS train:\n(omitting CP and sprinkles or preposing as not supported in the grammar of this Restricted Access Sequence Processing Transformer equivalent program yet)\n{mean_em_score*100}% or {num_right} out of {len(sentences)}")
print("\n\n\n")

mean_semantic_exact_match_score, num_right_semantic_exact_match, semantic_exact_matches = get_semantic_exact_match_score(lfs_computed, lfs_true)
print("\n\n\n")
if score_on_train_sample:
  print(f"Semantic match score on first {len(sentences)} of ReCOGS_pos train:\n(omitting CP and sprinkles or preposing as not supported in the grammar of this Restricted Access Sequence Processing Transformer equivalent program yet)\n{mean_semantic_exact_match_score*100}% or {num_right_semantic_exact_match} out of {len(sentences)}")
else:
  print(f"Semantic match score on the {len(sentences)} of ReCOGS_pos dev:\n(omitting CP as not supported in the grammar of this Restricted Access Sequence Processing Transformer equivalent program yet)\n{mean_semantic_exact_match_score*100}% or {num_right_semantic_exact_match} out of {len(sentences)}")

semantic_mismatches = []
for idx in range(len(semantic_exact_matches)):
  if semantic_exact_matches[idx] == 0:
    semantic_mismatches.append((lfs_computed[idx], lfs_true[idx]))
print(f"Mismatches: {semantic_mismatches}")

# quit RASP
input_lines = [bytes("quit()\n", 'utf8')]
stdin_handle.writelines(input_lines)
stdin_handle.flush()
stdin_handle.close()
stdout_handle.close()
proc.kill()
