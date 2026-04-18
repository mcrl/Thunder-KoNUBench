from datasets import Dataset
import random

def process_docs_symbol(dataset) -> Dataset:

  dataset = Dataset.from_list(dataset)

  def _process_doc(doc):
    std = doc.get("standard_negation", "")
    local = doc.get("local_negation", "")
    contra = doc.get("contradiction", "")
    para = doc.get("paraphrase", "")

    # instruction
    """
    standard negation:
    - 주절의 서술어를 한국어의 부정 표현을 활용해 부정함으로써 원문 P를  ¬P로 만든다.
    - 주절의 서술어 외의 나머지 부분은 수정하지 않는다.
    - 원문이 조건문일 때는 논리적 규칙(¬(P → Q) ≡ P ∧ ¬Q)을 따라 부정한다.
    - 주절이 여러 개일 경우 드모르간의 법칙(예. ¬(P ∧ Q) ≡ ¬P ∨ ¬Q)에 따라 모든 주절의 서술어를 부정한다.

    한국어의 부정 표현:
    - 안 계열: 안, -지 않다
    - 못 계열: 못, -지 못하다
    - 말다
    - 어휘적 부정: 상보 반의어를 활용한 부정(이다/아니다, 참석하다/불참하다 등)

    문제: 다음의 원문을 standard negation을 활용하여 올바르게 부정한 문장을 고르시오.
    원문: {doc['original_sentence']}

    A. {doc['standard_negation']}
    B. {doc['local_negation']}
    C. {doc['contradiction']}
    D. {doc['paraphrase']}

    정답:
    """

    prompt = f"문제: 다음의 원문을 standard negation을 활용하여 올바르게 부정한 문장을 고르시오.\n원문: {doc['original_sentence']}\n"
    if (local == None) or (local == ""):
      choices_str = [std, contra, para]
      choices = ["A", "B", "C"]
    else:
      choices_str = [std, local, contra, para]
      choices = ["A", "B", "C", "D"]
    
    idx = doc["idx"]
    if idx.startswith("G"):
       idx = idx[1:]

    random.seed(int(idx))
    random.shuffle(choices_str)
    gold = choices_str.index(std)

    for i in range(len(choices)):
      prompt += f"{choices[i]}. {choices_str[i]}\n"
    prompt += "\n정답:"

    return {"query": prompt,
            "choices": choices,
            "gold": gold}

  return dataset.map(_process_doc)
