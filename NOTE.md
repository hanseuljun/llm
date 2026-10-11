# bigram

started with a bigram model. hit about 40% accuracy.
learned these i guess?
- claude is good at making you a dataset.
- mlx syntax is pretty similar to pytorch.

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

