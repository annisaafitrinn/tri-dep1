# TRI-DEP: A Trimodal Comparative Study for Depression Detection Using Speech, Text, and EEG

**Eleonora Mancini\*, Annisaa Fitri Nurfirdausi\*, Paolo Torroni**

DISI, University of Bologna, Italy

\* Equal contribution

[![arXiv](https://img.shields.io/badge/arXiv-2510.14922-b31b1b.svg)](https://arxiv.org/abs/2510.14922)

---

## 📄 Abstract

Depression is a widespread mental health disorder, yet its automatic detection remains challenging. Prior work has explored unimodal and multimodal approaches, with multimodal systems showing promise by leveraging complementary signals. However, existing studies are limited in scope, lack systematic comparisons of features, and suffer from inconsistent evaluation protocols. We address these gaps by systematically exploring feature representations and modelling strategies across EEG, together with speech and text. We evaluate handcrafted features versus pre-trained embeddings, assess the effectiveness of different neural encoders, compare unimodal, bimodal, and trimodal configurations, and analyse fusion strategies with attention to the role of EEG. Consistent subject-independent splits are applied to ensure robust, reproducible benchmarking. Our results show that (i) the combination of EEG, speech and text modalities enhances multimodal detection, (ii) pretrained embeddings outperform handcrafted features, and (iii) carefully designed trimodal models achieve state-of-the-art performance. Our work lays the groundwork for future research in multimodal depression detection.

---

## 🔗 Links

- 📝 **Paper (arXiv):** https://arxiv.org/abs/2510.14922
- 🌐 **Project website:** https://annisaafitrinn.github.io/tri-dep/
- 💻 **Code repository:** https://github.com/annisaafitrinn/tri-dep1

---

## 🗂️ About this repository

This repository hosts the **project website** for the TRI-DEP paper. The site is a static HTML page built on the [Nerfies](https://nerfies.github.io/) template.

The full implementation — preprocessing, feature extraction, model training, and fusion scripts — lives in the **[tri-dep1](https://github.com/annisaafitrinn/tri-dep1)** repository.

---

## 📖 Citation

If you find this work useful, please cite:

```bibtex
@article{nurfirdausi2025tridep,
  title     = {TRI-DEP: A Trimodal Comparative Study for Depression Detection Using Speech, Text, and EEG},
  author    = {Nurfirdausi, Annisaa Fitri and Mancini, Eleonora and Torroni, Paolo},
  journal   = {arXiv preprint arXiv:2510.14922},
  year      = {2025}
}
```

---

## 🙏 Acknowledgments

Parts of this project page were adopted from the [Nerfies](https://nerfies.github.io/) page.

## 📜 Website License

This work is licensed under a [Creative Commons Attribution-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-sa/4.0/).
