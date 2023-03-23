import multiprocessing

# Get the number of CPUs available on your machine
n_cpus = multiprocessing.cpu_count()

print("Number of available CPUs: ", n_cpus)