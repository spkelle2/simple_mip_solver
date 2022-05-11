# how close to an integer a value must be to be considered integer
variable_epsilon = 1e-4

# threshold for saying fractional coef is good approximation
good_coefficient_approximation_epsilon = 1e-2

# threshold for considering two fractions are the same
exact_coefficient_approximation_epsilon = 1e-14

# threshold for considering an invalid cut valid
cut_tolerance = 1e-14

# maximum number of nonzero coefficients in allowable cut - can be high if no integer coef reqs
max_nonzero_coefs = 1000000

# number of degrees two cuts are within to be considered too parallel
parallel_cut_tolerance = 10

# relative improvement cutting plane algorithm must make to continue
cutting_plane_progress_tolerance = .001

max_cut_generation_iterations = 10

# largest allowed ratio of absolute values of cut coef to root LP relaxation coef
max_relative_cut_term_ratio = 1000

# minimum euclidean distance between cut and relaxation solution to add cut to model
min_cut_depth = 1e-8

# smallest acceptable norm for disjunctive cut
min_cglp_norm = 1e-4