"""
Tools used to run the Monte Carlo simulations of the Apples wit Apples paper.
This code is only needed to reproduce the results of the paper.
"""
import numpy as np
from multiprocessing import Pool, shared_memory


def draw_noise(distribution,
               num_draws,
               num_noise_observations_in,
               loc_noise,
               scale_noise):
    """
    Simple function to draw noise for the MC simulation

    Args:
        distribution: type of the noise: gaussian / laplace
        num_draws: number of noise samples
        num_noise_observations_in: sample size i.e. number of noise values
            at a given separation
        loc_noise: The location of the noise i.e. the center of the pdf.
        scale_noise: The width of the distribution. In case of gaussian noise
            this is the std. For the laplace this is b.

    Returns: np.array containing the noise

    """
    if distribution == "gaussian":
        return np.random.normal(loc=loc_noise, scale=scale_noise,
                                size=(num_draws, num_noise_observations_in))
    else:
        return np.random.laplace(loc=loc_noise, scale=scale_noise,
                                 size=(num_draws, num_noise_observations_in))


def draw_mp(shared_memory_parameters,
            num_sub_draws,
            distribution,
            num_noise_observations_in,
            idx,
            loc_noise,
            scale_noise):
    """
    Function to sample noise using multiprocessing
    Args:
        shared_memory_parameters: tuple containing:
            (the name of the shared memory, the shared memory size)
        num_sub_draws: number of samples to draw
        distribution: type of the noise: gaussian / laplace
        num_noise_observations_in:  sample size i.e. number of noise values
            at a given separation
        idx: index of the multiprocessing experiment. Needed to tell the
            subprocess where to store the results within the shared memory
        loc_noise: The location of the noise i.e. the center of the pdf.
        scale_noise: The width of the distribution. In case of gaussian noise
            this is the std. For the laplace this is b.

    Returns: None

    """
    # setup link to the shared memory
    shared_memory_name, memory_size = shared_memory_parameters

    shared_np_array = shared_memory.SharedMemory(name=shared_memory_name)
    np_array = np.ndarray(memory_size,
                          dtype=np.float64,
                          buffer=shared_np_array.buf)

    # Set a new random seed to get different random values for
    # every parallel process
    np.random.seed()

    # Draw random values
    result = draw_noise(distribution,
                        num_sub_draws,
                        num_noise_observations_in,
                        loc_noise=loc_noise,
                        scale_noise=scale_noise)

    # Store the result
    np_array[idx * num_sub_draws: (idx + 1) * num_sub_draws, :] = result
    print(".", end='')


def draw_mc_sample(num_noise_observations_in,
                   num_draws=1,
                   noise_distribution="gaussian",
                   loc_noise=0,
                   scale_noise=1,
                   num_cores=1):
    """
    Function for efficient noise sampling in the MC simulation. If more than
    10e6 noise values are requested multiprocessing will be used.

    Args:
        num_noise_observations_in: sample size i.e. number of noise values
            at a given separation
        num_draws: Number of monte carlo experiments to run.
        noise_distribution: type of the noise: gaussian / laplace
        loc_noise: The location of the noise i.e. the center of the pdf.
        scale_noise: The width of the distribution. In case of gaussian noise
            this is the std. For the laplace this is b.
        num_cores: number of CPU cores to use i.e. parallel processes.

    Returns:
        planet_observation: np.array containing (num_draws) noise values
        noise_observation: np.array containing (num_draws,
            num_noise_observations_in) noise values
        shared_np_array: if num_draws > 10e6: instance of the shared memory used
            during multiprocessing. Else None.
    """

    # make sure num_draws is an int
    num_draws = int(num_draws)

    if num_draws >= 10e6:
        # For large noise generation we use multiprocessing
        print("Generate random noise:")

        # Create a shared memory for the results.
        # This is needed to keep the memory usage small
        # 8 is the byte size of float64, +1 for the planet observations
        shared_np_array = shared_memory.SharedMemory(
            create=True, size=(num_draws * (num_noise_observations_in + 1) * 8))

        memory_size = (num_draws, num_noise_observations_in + 1,)
        shared_memory_name = shared_np_array.name
        memory_parameters = (shared_memory_name, memory_size)
        np_array = np.ndarray(memory_size, dtype=np.float64,
                              buffer=shared_np_array.buf)

        # Create the multiprocessing Pool
        pool = Pool(processes=num_cores)

        if num_draws % 100 != 0:
            raise ValueError("Only num_draws dividable by 100 are supported")

        # We create 100 subtasks. Every subtask samples 1/100 of the total noise
        # needed for the MC simulation
        draws_per_task = int(num_draws / 100)
        pool.starmap(draw_mp, [(memory_parameters,
                                int(draws_per_task),
                                noise_distribution,
                                int(num_noise_observations_in + 1),
                                j,
                                loc_noise,
                                scale_noise) for j in range(100)])

        pool.close()

        # copy the results from the buffer
        planet_observation = np_array[:, 0].reshape(num_draws)
        noise_observation = np_array[:, 1:]

        print("[DONE]")

    else:
        # no need to use multiprocessing
        shared_np_array = None
        planet_observation = draw_noise(noise_distribution,
                                        num_draws,
                                        1,
                                        loc_noise=loc_noise,
                                        scale_noise=scale_noise).flatten()

        noise_observation = draw_noise(noise_distribution,
                                       num_draws,
                                       num_noise_observations_in,
                                       loc_noise=loc_noise,
                                       scale_noise=scale_noise)

    if num_draws == 1:
        # in case of a single noise observation we return flattened np.arrays
        noise_observation = noise_observation[0]
        planet_observation = planet_observation[0]

    return planet_observation, noise_observation, shared_np_array
