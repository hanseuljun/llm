# bigram

started with a bigram model. hit about 40% accuracy.
learned these i guess?
- claude is good at making you a dataset.
- mlx syntax is pretty similar to pytorch.

pred_words: ['<bos>', 'the', 'dog', '<eos>']

# bow-v2

just doing word embedding with positional embedding already does something. hit accuracy of about 56%.
using padding and masking to have batching does make things run much faster, even in this size.
and the mlx's way of getting gradients is pretty straightforward.
i like how they are explicitly separate variables.
and the tensor suffix is pretty neat, totatlly helps reading code.

generated_words: ['<bos>', 'the', 'cat', 'near', 'the', 'cat', 'sees', 'the', 'cat', '<eos>']

# v3 data
getting the right dataset to push the model forward itself is a thing itself.
the dataset did not have a clear signal on "when a verb shows up" for a BOW model to predict.

# bow-v4

this one was based on the v4 data, which was an improvement on v3 data.
it contained a clear signal on "when a verb shows up".
the problem was, it was too easy for the BOW model with positional embedding.
this BOW model was already hitting 99% accuracy!
made me think it is time to move on to the next step and finally do something kind of transformer.

generated_words: ['<bos>', 'at', 'the', 'garden', 'the', 'mouse', 'by', 'the', 'mouse', 'by', 'the', 'mouse', 'chases', 'the', 'mouse', '<eos>']

# self-attention-v5

this time, the dataset became much more sharply aiming the target. associative recall, something the BOW model clearly fails.
goal is to see "g 2 s 5 b 11 m 1 r 5 d 7 ? r" and return 5, since 5 is after r.
BOW can't and didn't cut it so time for some attention.
in short, one self attention layer did not cut it.
just one step of mapping a token to another token is just not complex enough to figure out the relationship needed to do associative recall.
it kind of makes sense if you think about it.
and this one goes deep about this: https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads
accuracy was stuck at 5% and it was time to add at least another attention layer.
did causal and padding masking by adding -inf to the QK matrix, which was new to me!
and i could feel the FlashAttention becoming some next step.

output_str: ['<bos>', 'e', '19', 'm', '6', 'a', '9', 'i', '18', 'm', '17', 'q', '11', 's', '0', 'n', '15', '?', 'q', '10', '<eos>']

# transformer-v5

not really a transformer since no multi-headed attention, but another attention layer added.
impressively this one hit 99% accuracy for associative recall.
and interestingly, this accuracy only happens when the two attention layers and the two MLPs share weight.
if the two attention layers have their own weights, it breaks down. same for the MLPs.
in my limited experience, i haven't seen this sort-of regularization to have this dramatic of an effect but separate weights drops accuracy down to 30%.
also, some seeds make the accuracy again go to 30%.
seems like this is because, with separate weights, the model learns some wrong short-term path to lower the loss?
and kind of matches the looped transformer thing with the newest LLMs? surprisingly my toy model leads to a similar idea to them??
overall, these tiny things can break down the training process.
and seems like decoding strategy can be a thing in the next steps.
and do the batching as much as i can massively helps iteration.

output_str: ['<bos>', 't', '7', 'p', '2', 'l', '1', 'd', '11', 'n', '9', 'm', '15', 'a', '0', '?', 'j', '9', '<eos>']
