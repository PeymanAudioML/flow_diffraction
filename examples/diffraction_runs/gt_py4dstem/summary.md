# Ground truth: ours (skimage peak_local_max) vs py4DSTEM get_maxima_2D on dataPots
1875 patterns (25 materials x 3 angles x thicknesses); matching one-to-one within 3 px; direct beam removed from all sets

## given {'minSpacing': 4, 'maxNumPeaks': 100}
ours 529766 spots (510088 visible); py4DSTEM 185835; matched 178497 = 33.7% of ours (33.7% within 1 px); only ours 351269 (visible 334775 = 65.6% of visible labels); only py4DSTEM 7338
       n_ours   n_vis  given_n  given_matched  given_only_ours_vis  given_only_py4d  matched_frac  vis_missing_frac
angle                                                                                                              
1.0    176365  162254    61897          59465               105223             2432         0.337             0.649
2.0    177079  172762    61921          59452               113967             2469         0.336             0.660
4.0    176322  175072    62017          59580               115585             2437         0.338             0.660
patterns where py4DSTEM hit maxNumPeaks=100 (after beam removal >= 99): 99.6%

## nocap {'minSpacing': 4, 'maxNumPeaks': 100000}
ours 529766 spots (510088 visible); py4DSTEM 802532; matched 528553 = 99.8% of ours (99.6% within 1 px); only ours 1213 (visible 976 = 0.2% of visible labels); only py4DSTEM 273979
       n_ours   n_vis  nocap_n  nocap_matched  nocap_only_ours_vis  nocap_only_py4d  matched_frac  vis_missing_frac
angle                                                                                                              
1.0    176365  162254   267166         175943                  303            91223         0.998             0.002
2.0    177079  172762   268615         176682                  322            91933         0.998             0.002
4.0    176322  175072   266751         175928                  351            90823         0.998             0.002

## nocap_s3 {'minSpacing': 3, 'maxNumPeaks': 100000}
ours 529766 spots (510088 visible); py4DSTEM 875959; matched 528993 = 99.9% of ours (99.7% within 1 px); only ours 773 (visible 542 = 0.1% of visible labels); only py4DSTEM 346966
       n_ours   n_vis  nocap_s3_n  nocap_s3_matched  nocap_s3_only_ours_vis  nocap_s3_only_py4d  matched_frac  vis_missing_frac
angle                                                                                                                          
1.0    176365  162254      291379            176099                     148              115280         0.998             0.001
2.0    177079  172762      293249            176820                     189              116429         0.999             0.001
4.0    176322  175072      291331            176074                     205              115257         0.999             0.001

## position offsets py4DSTEM (nocap, sub-pixel) - ours (whole pixels), matched pairs
        dy                 dx              
      mean median    std mean median    std
angle                                      
1.0    0.0    0.0  0.248  0.0    0.0  0.244
2.0    0.0    0.0  0.248  0.0    0.0  0.246
4.0    0.0    0.0  0.246  0.0    0.0  0.244
|offset|: median 0.310 px, 95 % 0.530 px, max 2.995 px
nocap: unmatched ours, median dataPots / max 0.023; unmatched py4DSTEM, median intensity / max 0.015