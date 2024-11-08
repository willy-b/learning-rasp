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
argparser.add_argument("--use_gen_split", action="store_true")
argparser.add_argument("--use_test_split", action="store_true")
argparser.add_argument("--cp_examples_only", action="store_true")
# if Google Colab or other environment crashes, you may want to pick up where it left off
argparser.add_argument("--skip_rows", type=int, default=0)
argparser.add_argument("--do_pp_recursion_gen_split", action="store_true") # needs to be done separately as very slow
args = argparser.parse_args()
if (args.use_gen_split and args.use_dev_split) or (args.use_gen_split and args.use_test_split) or (args.use_dev_split and args.use_test_split):
  print("Please select just one of the arguments `--use_gen_split`,`--use_dev_split`, or `--use_test_split`")
  raise Exception("Please select just one of the arguments `--use_gen_split`,`--use_dev_split`,`--use_test_split`")

if args.do_pp_recursion_gen_split and not args.use_gen_split:
  print("`do_pp_recursion_gen_split` can only be used when `--use_gen_split` is used!")
  raise Exception("`do_pp_recursion_gen_split` can only be used when `--use_gen_split` is used!")

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
os.chdir(base_path)

with open(base_path + "/" + "word-level-pos-tokens-recogs-style-decoder-loop.rasp", "r") as f:
  rasp_setup_lines = f.readlines()
input_lines = [bytes(line, 'utf8') for line in rasp_setup_lines]

stdin_handle.writelines(input_lines)
stdin_handle.flush()

# discard all output from running the setup part of the script
input_lines = ["output;\n", "autoregressive_output = output;\n"]
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

score_on_train_sample = not args.use_dev_split and not args.use_gen_split and not args.use_test_split

print(f"Fetching script for ReCOGS Semantic Exact Match scoring which is more flexible than exact string match (ignores irrelevant formatting differences and ordering) from Wu et al 2023's repo...")
if not os.path.exists(base_path + "/compgen.py"):
  os.chdir(base_path)
  subprocess.run("wget https://raw.githubusercontent.com/frankaging/ReCOGS/1b6eca8ff4dca5fd2fb284a7d470998af5083beb/utils/compgen.py", shell=True, executable='/bin/bash')
from compgen import recogs_exact_match
def semantic_exact_match(lf_predicted, lf_actual):
  return 1.0 if recogs_exact_match(lf_predicted.strip().lower().replace(" and ", " AND "),lf_actual.strip().lower().replace(" and ", " AND ")) else 0.0

def get_clopper_pearson_confidence_interval(n, k):
  alpha = 0.05
  from scipy.stats import beta
  cp_confidence_interval = beta.ppf([alpha/2.0, 1-alpha/2.0], [k, k+1],[n-k + 1, n-k])
  if n == k:
    cp_confidence_interval[1] = 1.0
  if k == 0:
    cp_confidence_interval[0] = 0.0
  return cp_confidence_interval

def get_semantic_exact_match_score(lfs_predicted, lfs_actual):
  # for semantic exact match we must NOT lowercase the "AND" as it searches for the pattern in a case-sensitive way
  semantic_exact_matches = np.array([semantic_exact_match(lfs_predicted[idx], lfs_actual[idx]) for idx in range(len(lfs_predicted))])
  mean_semantic_exact_match_score = semantic_exact_matches.mean()
  num_right_semantic_exact_match = semantic_exact_matches.sum()
  count = len(lfs_predicted)
  semantic_exact_match_score_confidence_interval = get_clopper_pearson_confidence_interval(count, num_right_semantic_exact_match)
  return mean_semantic_exact_match_score, num_right_semantic_exact_match, semantic_exact_matches, semantic_exact_match_score_confidence_interval

def get_percentages_with_ci_groupby_binary_data(df, groupby_key, alpha=0.05, print_result=False):
  dfgb = df.groupby(groupby_key)
  c = dfgb.count()
  s = dfgb.sum()
  c.columns = ["count"]
  ci_lows = {}
  ci_highs = {}
  p = {}
  for idx in c.index:
    n = c.loc[idx].values[0]
    k = s.loc[idx].values[0]
    from scipy.stats import beta
    ci = get_clopper_pearson_confidence_interval(n, k)
    ci_lows[idx] = ci[0]*100.0
    ci_highs[idx] = ci[1]*100.0
    p[idx] = float(k)/float(n)*100
    if print_result:
      print(f"{idx}: {p[idx]:0.2f}% ({(1-alpha)*100:0.2f}% confidence interval: {ci_lows[idx]:0.2f}% to {ci_highs[idx]:0.2f}% ({k} out of {n})")
  c.insert(0, "hits", s)
  c.insert(0, "percentage", p)
  c.insert(0, "percentage_ci_high", ci_highs)
  c.insert(0, "percentage_ci_low", ci_lows)
  return c

optional_cp_filter = "grep 'that' |" if args.cp_examples_only else ""
if score_on_train_sample:
  print("Now load official Wu et al 2023 ReCOGS training examples\n(sample from https://raw.githubusercontent.com/frankaging/ReCOGS/1b6eca8ff4dca5fd2fb284a7d470998af5083beb/recogs_positional_index/train.tsv , associated with https://arxiv.org/abs/2303.13716 )")
  # one of official author's dataset for ReCOGS paper
  subprocess.run("wget https://raw.githubusercontent.com/frankaging/ReCOGS/1b6eca8ff4dca5fd2fb284a7d470998af5083beb/recogs_positional_index/train.tsv", shell=True)
  subprocess.run("echo 'COGS Sentence	ReCOGS Logical Form	Distribution' > train_in_distribution_no_sprinkles.tsv", shell=True)
  subprocess.run(f"cat train.tsv | grep 'in_distribution' | {optional_cp_filter} grep -v 'sprinkle' >> train_in_distribution_no_sprinkles.tsv", shell=True)
  recogs_datafile = pd.read_csv("train_in_distribution_no_sprinkles.tsv", delimiter="	")
else:
  if args.use_dev_split:
    print("Using dev split, the `num_train_examples_to_check` argument will be ignored")
    print("Now load official Wu et al 2023 ReCOGS dev split (excluding complement phrases as not yet supported)\n(https://raw.githubusercontent.com/frankaging/ReCOGS/1b6eca8ff4dca5fd2fb284a7d470998af5083beb/recogs_positional_index/dev.tsv , associated with https://arxiv.org/abs/2303.13716 )")
    # one of official author's dataset for ReCOGS paper
    subprocess.run("wget https://raw.githubusercontent.com/frankaging/ReCOGS/1b6eca8ff4dca5fd2fb284a7d470998af5083beb/recogs_positional_index/dev.tsv", shell=True)
    subprocess.run("echo 'COGS Sentence	ReCOGS Logical Form	Distribution' > dev_with_header.tsv", shell=True)
    subprocess.run(f"cat dev.tsv {optional_cp_filter} >> dev_with_header.tsv", shell=True)
    recogs_datafile = pd.read_csv("dev_with_header.tsv", delimiter="	")
  elif args.use_test_split:
    print("Using test split, the `num_train_examples_to_check` argument will be ignored")
    print("Now load official Wu et al 2023 ReCOGS test split (excluding complement phrases as not yet supported)\n(https://raw.githubusercontent.com/frankaging/ReCOGS/1b6eca8ff4dca5fd2fb284a7d470998af5083beb/recogs_positional_index/test.tsv , associated with https://arxiv.org/abs/2303.13716 )")
    # one of official author's dataset for ReCOGS paper
    subprocess.run("wget https://raw.githubusercontent.com/frankaging/ReCOGS/1b6eca8ff4dca5fd2fb284a7d470998af5083beb/recogs_positional_index/test.tsv", shell=True)
    subprocess.run("echo 'COGS Sentence	ReCOGS Logical Form	Distribution' > test_with_header.tsv", shell=True)
    subprocess.run(f"cat test.tsv {optional_cp_filter} >> test_with_header.tsv", shell=True)
    recogs_datafile = pd.read_csv("test_with_header.tsv", delimiter="	")
  elif args.use_gen_split:
    print("Using gen split, the `num_train_examples_to_check` argument will be ignored")
    print("Now load official Wu et al 2023 ReCOGS gen split (excluding complement phrases as not yet supported)\n(https://raw.githubusercontent.com/frankaging/ReCOGS/1b6eca8ff4dca5fd2fb284a7d470998af5083beb/recogs_positional_index/gen.tsv , associated with https://arxiv.org/abs/2303.13716 )")
    filename = "gen_only_pp_recursion.tsv" if args.do_pp_recursion_gen_split else "gen_no_pp_recursion.tsv"
    # one of official author's dataset for ReCOGS paper
    subprocess.run("wget https://raw.githubusercontent.com/frankaging/ReCOGS/1b6eca8ff4dca5fd2fb284a7d470998af5083beb/recogs_positional_index/gen.tsv", shell=True)
    subprocess.run(f"echo 'COGS Sentence	ReCOGS Logical Form	Distribution' > {filename}", shell=True)
    if not args.do_pp_recursion_gen_split:
      print("Note pp recursion split (which is slow) is left out by default, run `--do_pp_recursion_gen_split` to score that (it is supported)")
      subprocess.run(f"cat gen.tsv | {optional_cp_filter} grep -v 'pp_recursion' >> {filename}", shell=True)
      recogs_datafile = pd.read_csv("gen_no_pp_recursion.tsv", delimiter="	")
    else:
      print("Just assessing pp recursion split (which is slow)")
      subprocess.run(f"cat gen.tsv | {optional_cp_filter} grep 'pp_recursion' >> {filename}", shell=True)
      recogs_datafile = pd.read_csv(f"{filename}", delimiter="	")

pos_desc = "first"
if args.skip_rows > 0:
  pos_desc = f"first (after {args.skip_rows})"
  print(f"Skipped {args.skip_rows} per argument")
  recogs_datafile = pd.DataFrame(recogs_datafile.values[args.skip_rows:], columns=recogs_datafile.columns)

if score_on_train_sample:
  sentences = recogs_datafile["COGS Sentence"][:args.num_train_examples_to_check]
  lfs_true = recogs_datafile["ReCOGS Logical Form"][:args.num_train_examples_to_check]
  labels = recogs_datafile["Distribution"][:args.num_train_examples_to_check]
else:
  sentences = recogs_datafile["COGS Sentence"]
  lfs_true = recogs_datafile["ReCOGS Logical Form"]
  labels = recogs_datafile["Distribution"]

label = "test" if args.use_test_split else ("dev" if args.use_dev_split else ("gen" if args.use_gen_split else "data"))
sentences = [sentence.replace(" .", "").replace(".", "") for sentence in sentences]
lfs_computed = []
semantic_exact_matches = []
for idx in range(len(sentences)):
  try:
    output = process_example(sentences[idx], False)
  except:
    print(f"Could not process input '{sentences[idx]}'")
    output = ""
  lfs_computed.append(output)
  semantic_exact_matches.append(semantic_exact_match(output, lfs_true[idx]))
  n = len(semantic_exact_matches)
  k_sem_matches = np.array(semantic_exact_matches).sum()
  sem_pct = k_sem_matches/n*100.0
  sem_ci_pct = get_clopper_pearson_confidence_interval(n, k_sem_matches)*100.0
  disclaimer = "(omitting training data augmentations like sprinkles or preposing as not supported in the grammar of this Restricted Access Sequence Processing Transformer equivalent program and irrelevant to dev,test,gen sets)" if score_on_train_sample else ""
  print(f"Semantic exact match score on {pos_desc} {n} of ReCOGS_pos {label}:\n{disclaimer}\n{sem_pct:0.2f}% or {k_sem_matches} out of {n} (95% confidence interval: {sem_ci_pct[0]:0.2f}% to {sem_ci_pct[1]:0.2f}%)")
  if args.use_gen_split:
        gen_sem_df = pd.DataFrame([{"Semantic Exact Match": semantic_exact_matches[jdx], "Category": labels[jdx]} for jdx in range(idx+1)], columns=["Semantic Exact Match", "Category"])
        print("\n")
        print(f"Semantic Exact Match % by category:")
        sem_by_category_with_ci = get_percentages_with_ci_groupby_binary_data(gen_sem_df, "Category", print_result=True)
        print("\n")
  if idx % 10 == 0:
    # update CSV (these are small so can rewrite each time)
    output_df = pd.DataFrame([{"Input Sentence": sentences[jdx], "Logical Form Predicted": lfs_computed[jdx], "Label": labels[jdx]} for jdx in range(idx+1)], columns=["Input Sentence", "Logical Form Predicted","Label"])
    output_df.to_csv("lf_output.tsv", index=False, sep="	")

output_df = pd.DataFrame([{"Input Sentence": sentences[jdx], "Logical Form Predicted": lfs_computed[jdx]} for jdx in range(len(lfs_computed))], columns=["Input Sentence", "Logical Form Predicted"])
output_df.to_csv("lf_output.tsv", index=False, sep="	")

# note not finished with the grammar yet and it will, when scored by Exact Match, systematically miss the v_inf_taking_to_v_inf (e.g. "the scientist wanted to read") due to misordered output vs reference
# in ReCOGS "semantic exact match" they are still be correct as it ignores semantically meaningless reorderings but we can compute this also
exact_matches = [1.0 if lfs_computed[idx].strip().lower() == lfs_true[idx].strip().lower() else 0.0 for idx in range(len(lfs_computed))]
mean_em_score = np.array(exact_matches).mean()
num_em_right = np.array(exact_matches).sum()

if score_on_train_sample:
  print(f"Exact Match score on {pos_desc} {len(sentences)} of ReCOGS train:\n(omitting training data augmentations like sprinkles or preposing as not supported in the grammar of this Restricted Access Sequence Processing Transformer equivalent program and irrelevant to dev,test, or gen sets)\n{mean_em_score*100}% or {num_em_right} out of {len(sentences)}")
print("\n\n\n")

mean_semantic_exact_match_score, num_right_semantic_exact_match, semantic_exact_matches, semantic_exact_match_score_confidence_interval = get_semantic_exact_match_score(lfs_computed, lfs_true)
print("\n\n\n")
if score_on_train_sample:
  print(f"Semantic exact match score on {pos_desc} {len(sentences)} of ReCOGS_pos train:\n(omitting training data augmentations like sprinkles or preposing as not supported in the grammar of this Restricted Access Sequence Processing Transformer equivalent program and irrelevant to evaluation on dev,test,gen sets)\n{mean_semantic_exact_match_score*100:0.2f}% or {num_right_semantic_exact_match} out of {len(sentences)} (95% confidence interval: {semantic_exact_match_score_confidence_interval[0]*100:0.2f}% to {semantic_exact_match_score_confidence_interval[1]*100:0.2f}%)")
else:
  print(f"Semantic exact match score on {pos_desc} {len(sentences)} of ReCOGS_pos {label}:\n{mean_semantic_exact_match_score*100:0.2f}% or {num_right_semantic_exact_match} out of {len(sentences)} (95% confidence interval: {semantic_exact_match_score_confidence_interval[0]*100:0.2f}% to {semantic_exact_match_score_confidence_interval[1]*100:0.2f}%)")
  if args.use_gen_split:
    gen_sem_df = pd.DataFrame([{"Semantic Exact Match": semantic_match_scores[jdx], "Category": labels[jdx]} for jdx in range(len(lfs_computed))], columns=["Semantic Exact Match", "Category"])
    print("\n")
    print(f"Semantic Exact Match % by category:")
    sem_by_category_with_ci = get_percentages_with_ci_groupby_binary_data(gen_sem_df, "Category", print_result=True)
    print("\n")

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
