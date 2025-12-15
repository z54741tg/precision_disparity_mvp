# precision_disparity_mvp
Precision disparity (relative + absolute) on TP/FP “made links”

This mini-project is for testing and demonstrating a precision disparity approach over subgroups. The intended use is linkage evaluation where you have a dataset of made links (i.e., links your pipeline produced) and each made link has been classified as:

TP (true positive): a made link that is genuinely correct

FP (false positive): a made link that is genuinely incorrect

Given TP/FP labels, the project computes:

precision by subgroup

relative disparity ratio against a reference group (precision ratio)

absolute gap against a reference group (precision difference)

It also flags low-volume groups where rates are likely unstable.

What “precision” means here

Each row in the input dataframe represents a made link. For any subgroup:

tp = count of TP made links in that subgroup

fp = count of FP made links in that subgroup

linked = tp + fp (total made links in that subgroup)

precision = tp / (tp + fp)

This is a made-link view of precision (i.e. “of the links we made, what fraction were correct?”)
