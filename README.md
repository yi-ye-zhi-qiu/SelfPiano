## kitchen soundscapes

### Autoencoder-generated, neural network music based off my musical opinion of objects in my kitchen

![Metis logo](images/metis.png) Metis data-science bootcamp passion project, **Mar 01 - Mar 25 2021**

[See the final product](http://liamisaacs.com/kitchensoundscapes)

[See the blog post](l-teach-a-computer-to-reflect-the-sounds-of-a-space-218aa817bcc)

Project was presented, [slides!](final_presentation.pdf)

**Summary:**  Website view of a neural network's generation of music inspired by my own music about kitchen objects. The approach is to see music as an image, not a text. Input MIDI files are extracted for notes/chords, and an autoencoder is trained on that information.

----

Modules used:
- `music21`
- `jupyter notebook`
- `autoencoder`
- `lstm`
- other modules: `scikit-learn` `numpy`

----

The data:

- The dataset can be found in the "music" folder

----

The process:

- The basic idea is to use `load_songs.py` (or `get_songs` as is in the jupyter notebooks) to retrive notes/chords
- We then encode that using one-hot encoding (in the case of `basic_autoencoder.py` and `lstm.py`) or encode that into a sparse matrix as is in `autoencoder_with_sparse_matrix.py`
- We train an autoencoder/lstm on that data and generate a note based off a random starting point
- We can decode that back into notes and download it as a `.mid` file

----

On the web:

- The project can be found [here]((http://liamisaacs.com/kitchensoundscapes))
