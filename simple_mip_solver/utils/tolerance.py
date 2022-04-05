variable_epsilon = 1e-4  # how close to an integer a value must be to be considered integer
good_coefficient_approximation_epsilon = 1e-2  # threshold for saying fractional coef is good approximation
exact_coefficient_approximation_epsilon = 1e-14  # threshold for considering two fractions are the same
cut_tolerance = 1e-14  # threshold for considering an invalid cut valid
max_nonzero_coefs = 10  # maximum number of nonzero coefficients in allowable cut
parallel_cut_tolerance = 10  # number of degrees two cuts are within to be considered too parallel
cutting_plane_progress_tolerance = .001  # relative improvement cutting plane algorithm must make to continue
max_cut_generation_iterations = 10
