# optax-zclip
ZClip implementation for Optax.

## References

- Optax: [github](https://github.com/google-deepmind/optax), [docs](https://optax.readthedocs.io/en/latest/).
- ZClip: [paper](https://arxiv.org/abs/2504.02507), [github](https://github.com/bluorion-com/ZClip).


## Disclaimer

The name of the repository comes from the name of the technique, as given by the authors. However, this technique does not perform gradient clipping elementwise, but rather gradient scaling. This is similar to the kind of clipping used in [`optax.clip_by_global_norm`](https://optax.readthedocs.io/en/latest/api/transformations.html#optax.clip_by_global_norm), for instance.


## Building

Right now there is no release, please clone the repo and install it from source if needed.


## License

[MIT](https://choosealicense.com/licenses/mit/)