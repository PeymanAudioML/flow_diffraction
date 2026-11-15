# Simulated data: new 93 folders vs the 24 reference folders

probe radius -> angle (reference): {1.0: {'mean': 2.59, 'min': 2.59, 'max': 2.59}, 2.0: {'mean': 4.69, 'min': 4.69, 'max': 4.69}, 4.0: {'mean': 9.85, 'min': 9.85, 'max': 9.85}}

## new: 93 folders, 279 simulations, 6975 patterns, non-finite patterns 0
simulations per estimated angle: {1.0: 93, 2.0: 93, 4.0: 93}
         radius      n_gt     n_vis   spacing       eta  beam_share  dyn_range  beam_off
count  6975.000  6975.000  6975.000  6975.000  6975.000    6975.000   6975.000  6975.000
mean      5.708   280.338   229.369     8.590     1.522       0.664      6.340     1.509
std       3.054   239.959   182.141     4.102     0.975       0.281      0.786     4.422
min       2.585    22.000     5.000     4.000     0.156       0.000      4.274     0.000
10%       2.585   109.000    72.000     5.385     0.552       0.181      5.332     0.000
50%       4.687   208.000   186.000     7.280     1.231       0.749      6.316     0.000
90%       9.853   504.000   423.000    12.728     3.116       0.952      7.395     4.774
max       9.853  1584.000  1535.000    33.242     4.927       0.999      9.274    49.088

## ref: 24 folders, 72 simulations, 1800 patterns, non-finite patterns 0
simulations per estimated angle: {1.0: 24, 2.0: 24, 4.0: 24}
         radius      n_gt     n_vis   spacing       eta  beam_share  dyn_range  beam_off
count  1800.000  1800.000  1800.000  1800.000  1800.000    1800.000   1800.000  1800.000
mean      5.708   287.228   277.109     9.006     1.502       0.392      5.905     1.501
std       3.055   185.744   189.318     3.797     1.060       0.275      0.811     3.115
min       2.585    90.000    19.000     4.000     0.292       0.000      4.286     0.000
10%       2.585   117.000   111.000     5.000     0.517       0.066      4.887     0.000
50%       4.687   240.000   226.000     8.000     1.172       0.340      5.840     0.211
90%       9.853   560.200   550.200    14.318     3.078       0.834      6.967     4.189
max       9.853  1092.000  1092.000    17.692     4.927       0.998      9.183    22.087

identical dataPots shared with reference: 0; new hashes used by >1 folder: 0

eta regimes (share of patterns):
         separate  touching  overlap
dataset                             
new         0.320     0.353    0.327
ref         0.337     0.356    0.307