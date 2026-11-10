import sys, random
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("models/openai-community--gpt2")
random.seed(123)
# single-token words with a leading space, so repeating the text re-tokenises identically
cands = [i for i in range(1000, 50000) if (s := tok.decode([i])).startswith(" ") and s[1:].isalpha() and s[1:].isascii()]
lines = []
for _ in range(24):
    ids = random.sample(cands, 30)
    text = tok.decode(ids + ids)
    assert tok(text)["input_ids"] == ids + ids
    lines.append(text)
nat = [
 "The harbour was quiet that morning, and the fishing boats had already left before the sun came over the hills. An old man sat on the wall mending a net, humming to himself, while two children chased a dog along the wet sand.",
 "In 1984 the company moved its headquarters to Chicago, where it employed more than two thousand people. Sales rose steadily during the following decade, although profits were squeezed by rising costs and by competition from cheaper imports.",
 "To make the sauce, melt the butter in a small pan and stir in the flour. Add the milk a little at a time, whisking constantly so that no lumps form, then season with salt, pepper and a pinch of nutmeg.",
 "Scientists have long known that the climate of the region changed sharply at the end of the last ice age. New evidence from lake sediments suggests that the change happened within a few decades rather than over several centuries.",
 "She opened the letter slowly, afraid of what it might say. It was from her brother, who had not written for almost three years, and the first line told her that he was coming home in the spring.",
 "The match was played in heavy rain, and neither team managed to score in the first half. After the break the visitors took control, and a late goal from their captain gave them a win that few had expected.",
 "The committee will meet again next month to consider the proposal. Members said they needed more time to study the figures, and several asked whether the plan could be paid for without raising local taxes.",
 "Most birds in the valley build their nests in early April. The eggs hatch after about two weeks, and both parents feed the young until they are strong enough to leave the nest and find food for themselves.",
]
open(sys.argv[1], "w").write("\n".join(lines + nat) + "\n")
