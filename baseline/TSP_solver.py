import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
import time
from typing import Dict, List, Tuple, Optional

class TSPSolver:
    """
    Traveling Salesman Problem solver using Genetic Algorithm
    Supports both 2D and 3D coordinates
    """
    
    def __init__(self, pop_size: int = 150, max_iter: int = 4000, 
                 mutation_rate: float = 0.1, verbose: bool = True):
        """
        Initialize TSP Solver
        
        Parameters:
        pop_size: Population size for genetic algorithm
        max_iter: Maximum number of iterations
        mutation_rate: Mutation probability
        verbose: Whether to print progress
        """
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.mutation_rate = mutation_rate
        self.verbose = verbose
        self.best_route = None
        self.best_distance = float('inf')
        self.fitness_history = []
        self.is_3d = False
        
    def detect_dimension(self, cities: Dict) -> bool:
        """
        Detect if cities are in 2D or 3D
        
        Parameters:
        cities: Dictionary with city data
        
        Returns:
        True if 3D, False if 2D
        """
        first_city = list(cities.values())[0]
        coord = first_city['coord']
        return len(coord) == 3

    def pmx_crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[List, List]:
        """
        Partially Mapped Crossover (PMX) for TSP route crossover
        
        Parameters:
            parent1, parent2: Two parent routes (permutations)
        
        Returns:
            child1, child2: Two offspring routes
        """
        size = len(parent1)
        
        # Randomly select two crossover points
        crossover_points = np.random.choice(range(size), 2, replace=False)
        start, end = min(crossover_points), max(crossover_points)
        
        # Initialize children
        child1 = [None] * size
        child2 = [None] * size
        
        # Copy middle segment
        child1[start:end+1] = parent2[start:end+1]
        child2[start:end+1] = parent1[start:end+1]
        
        # Mapping relationships
        mapping1 = {parent2[i]: parent1[i] for i in range(start, end+1)}
        mapping2 = {parent1[i]: parent2[i] for i in range(start, end+1)}
        
        # Fill remaining positions
        for i in list(range(0, start)) + list(range(end+1, size)):
            # Handle child1
            value = parent1[i]
            while value in child1:
                value = mapping1.get(value, value)
            child1[i] = value
            
            # Handle child2
            value = parent2[i]
            while value in child2:
                value = mapping2.get(value, value)
            child2[i] = value
        
        return child1, child2

    def swap_mutation(self, route: np.ndarray) -> np.ndarray:
        """Swap mutation"""
        route = route.copy()
        i, j = np.random.choice(len(route), 2, replace=False)
        route[i], route[j] = route[j], route[i]
        return route

    def translate_indices(self, indices: np.ndarray, cities: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert city indices to coordinate arrays (supports both 2D and 3D)
        
        Parameters:
        cities: City dictionary with format {'index': {'coord': np.array([x,y]) or np.array([x,y,z]), 'name': str}, ...}
        indices: City index array
        
        Returns:
        Coordinate array and name array
        """
        indices = np.asarray(indices)
        original_shape = indices.shape
        
        flattened_indices = indices.reshape(-1)
        
        coord_list = []
        name_list = []
        for idx in flattened_indices:
            city_data = cities[idx]
            coord_list.append(city_data['coord'])
            name_list.append(city_data['name'])
        
        coord_array = np.array(coord_list).reshape(*original_shape, -1)
        name_array = np.array(name_list).reshape(original_shape)
        
        return coord_array, name_array

    def calculate_distance(self, city_seq: np.ndarray) -> np.ndarray:
        """
        Calculate Euclidean distances between adjacent cities in the sequence
        Works for both 2D and 3D coordinates
        
        Parameters:
        city_seq: Coordinate array, can be:
            - 2D case: (M, 2) for single route or (N, M, 2) for multiple routes
            - 3D case: (M, 3) for single route or (N, M, 3) for multiple routes
        
        Returns:
        Distance array with same dimensionality structure
        """
        if city_seq.ndim == 2:  # Single route (M, coord_dim)
            city_next = np.roll(city_seq, -1, axis=0)
            deltas = city_seq - city_next # todo  check
            dist = np.sqrt(np.sum(deltas**2, axis=1))
        elif city_seq.ndim == 3:  # Multiple routes (N, M, coord_dim)
            city_next = np.roll(city_seq, -1, axis=1)
            deltas = city_seq[:, :-1, :] - city_next[:, :-1, :]
            dist = np.sqrt(np.sum(deltas**2, axis=2))
        else:
            raise ValueError("city_seq must be (M, coord_dim) or (N, M, coord_dim) array")
        return dist

    def calculate_fitness(self, population: List[np.ndarray], cities: Dict) -> np.ndarray:
        """Calculate fitness for population"""
        pop_coords, _ = self.translate_indices(population, cities)
        pop_dists = self.calculate_distance(pop_coords)
        total_dists = np.sum(pop_dists, axis=1, keepdims=True)
        return total_dists

    def genetic_algorithm_step(self, population: List[np.ndarray], fitness: np.ndarray) -> List[np.ndarray]:
        """One step of genetic algorithm"""
        pop_size = len(population)
        fitness = fitness.flatten()
        
        # Selection - roulette wheel selection with fitness inversion
        # Higher fitness = lower distance = better solution
        max_dist = np.max(fitness)
        min_dist = np.min(fitness)
        adjusted_fitness = max_dist + min_dist/100 - fitness  # Invert fitness
        
        # Ensure non-negative fitness
        adjusted_fitness = np.maximum(adjusted_fitness, 1e-10)
        
        selected_indices = np.random.choice(
            pop_size, size=pop_size,
            p=adjusted_fitness/adjusted_fitness.sum()
        )
        offspring = [population[i] for i in selected_indices]
        offspring = np.random.permutation(offspring)

        # Crossover
        for i in range(0, pop_size, 2):
            if i+1 < pop_size:
                child1, child2 = self.pmx_crossover(offspring[i], offspring[i+1])
                offspring[i] = np.array(child1)
                offspring[i+1] = np.array(child2)
        
        # Mutation
        for i in range(pop_size):
            if np.random.rand() < self.mutation_rate:
                offspring[i] = self.swap_mutation(offspring[i])
        
        return offspring

    def solve(self, cities: Dict) -> Dict:
        """
        Solve TSP using genetic algorithm
        
        Parameters:
        cities: Dictionary with city data
        
        Returns:
        Dictionary containing solution details
        """
        start_time = time.time()
        num_cities = len(cities)
        
        # Detect if we're working with 2D or 3D coordinates
        self.is_3d = self.detect_dimension(cities)
        coord_dim = 3 if self.is_3d else 2
        
        if self.verbose:
            print(f"Solving TSP with {num_cities} cities in {'3D' if self.is_3d else '2D'} space...")
            print(f"Population size: {self.pop_size}, Max iterations: {self.max_iter}")
        
        # Initialize population
        population = [np.random.permutation(num_cities) for _ in range(self.pop_size)]
        
        self.fitness_history = []
        best_distances = []
        
        # Evolution loop
        for iteration in range(self.max_iter):
            # Calculate fitness
            distances = self.calculate_fitness(population, cities)
            fitness_scores = np.max(distances) + np.min(distances)/100 - distances
            
            # Track best solution
            best_idx = np.argmin(distances.flatten())
            current_best_distance = distances[best_idx, 0]
            current_best_route = population[best_idx].copy()
            
            best_distances.append(current_best_distance)
            
            # Update global best
            if current_best_distance < self.best_distance:
                self.best_distance = current_best_distance
                self.best_route = current_best_route
            
            if self.verbose and (iteration % 200 == 0 or iteration == self.max_iter - 1):
                print(f"Iteration {iteration}: Best distance = {current_best_distance:.2f}")
            
            # Generate new population
            population = self.genetic_algorithm_step(population, distances)
        
        end_time = time.time()
        solve_time = end_time - start_time
        
        if self.verbose:
            print(f"\nOptimization completed!")
            print(f"Best distance: {self.best_distance:.2f}")
            print(f"Time spent: {solve_time:.2f} seconds")
        
        # Get city names for the best route
        _, city_names = self.translate_indices(self.best_route, cities)
        
        return {
            'best_route_indices': self.best_route,
            'best_route_names': city_names,
            'best_distance': self.best_distance,
            'solve_time': solve_time,
            'fitness_history': best_distances,
            'cities': cities,
            'is_3d': self.is_3d
        }

    def plot_solution(self, solution: Dict, save_path, figsize: Tuple[int, int] = (15, 6)):
        """
        Plot the TSP solution (supports both 2D and 3D)
        
        Parameters:
        solution: Solution dictionary from solve() method
        figsize: Figure size tuple
        """
        is_3d = solution['is_3d']
        
        if is_3d:
            fig = plt.figure(figsize=figsize)
            ax1 = fig.add_subplot(121, projection='3d')
            ax2 = fig.add_subplot(122)
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot route visualization
        cities = solution['cities']
        route_indices = solution['best_route_indices']
        
        # Get coordinates for the route
        route_coords, route_names = self.translate_indices(route_indices, cities)
        
        # Close the loop by adding the first city at the end
        route_coords_closed = np.vstack([route_coords, route_coords[0]])

        # Get all city coordinates for plotting
        all_coords = np.array([cities[i]['coord'] for i in range(len(cities))])
        
        if is_3d:
            # 3D plotting
            ax1.scatter(all_coords[:, 0], all_coords[:, 1], all_coords[:, 2], 
                       c='red', s=100, alpha=0.8, label='Cities')
            
            # Plot route
            ax1.plot(route_coords[:, 0], route_coords[:, 1], route_coords[:, 2],
                    'b-', linewidth=2, alpha=0.7, label='Route')
            ax1.scatter(route_coords[:, 0], route_coords[:, 1], route_coords[:, 2],
                       c='blue', s=80, alpha=0.7)
            
            # Add city labels
            for i, coord in enumerate(route_coords):
                ax1.text(coord[0], coord[1], coord[2], route_names[i], fontsize=8)
            
            ax1.set_xlabel('X Coordinate')
            ax1.set_ylabel('Y Coordinate')
            ax1.set_zlabel('Z Coordinate')
            ax1.set_title(f'3D TSP Solution\nTotal Distance: {solution["best_distance"]:.2f}')
            ax1.legend()
            
        else:
            # 2D plotting
            ax1.scatter(all_coords[:, 0], all_coords[:, 1], c='red', s=100, zorder=5, alpha=0.8, label='Cities')
            
            # Plot route
            ax1.plot(route_coords_closed[:, 0], route_coords_closed[:, 1], 'b-', linewidth=2, alpha=0.7, label='Route')
            ax1.scatter(route_coords_closed[:-1, 0], route_coords_closed[:-1, 1], c='blue', s=80, alpha=0.7)
            
            # Add city labels
            for i, coord in enumerate(route_coords):
                ax1.annotate(route_names[i], (coord[0], coord[1]), 
                            xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            ax1.set_xlabel('X Coordinate')
            ax1.set_ylabel('Y Coordinate')
            ax1.set_title(f'2D TSP Solution\nTotal Distance: {solution["best_distance"]:.2f}')
            ax1.grid(True, alpha=0.3)
            ax1.axis('equal')
            ax1.legend()
        
        # Plot convergence history
        if solution['fitness_history']:
            ax2.plot(solution['fitness_history'], 'g-', linewidth=2)
            ax2.set_title('Convergence History')
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Best Distance')
            ax2.grid(True, alpha=0.3)
            
            # Add some statistics to the convergence plot
            final_distance = solution['fitness_history'][-1]
            initial_distance = solution['fitness_history'][0]
            improvement = ((initial_distance - final_distance) / initial_distance) * 100
            ax2.text(0.05, 0.95, f'Improvement: {improvement:.1f}%', 
                    transform=ax2.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    @staticmethod
    def create_random_cities_2d(num_cities: int, width: int = 800, height: int = 600) -> Dict:
        """
        Create a dictionary of random cities in 2D space
        
        Parameters:
        num_cities: Number of cities to generate
        width: Width of the area
        height: Height of the area
        
        Returns:
        Dictionary of cities with 2D coordinates
        """
        cities = {}
        for i in range(num_cities):
            cities[i] = {
                'coord': np.array([np.random.randint(0, width), np.random.randint(0, height)]),
                'name': f'City_{i}'
            }
        return cities

    @staticmethod
    def create_random_cities_3d(num_cities: int, width: int = 800, height: int = 600, depth: int = 400) -> Dict:
        """
        Create a dictionary of random cities in 3D space
        
        Parameters:
        num_cities: Number of cities to generate
        width: Width of the area (X dimension)
        height: Height of the area (Y dimension)  
        depth: Depth of the area (Z dimension)
        
        Returns:
        Dictionary of cities with 3D coordinates
        """
        cities = {}
        for i in range(num_cities):
            cities[i] = {
                'coord': np.array([
                    np.random.randint(0, width), 
                    np.random.randint(0, height),
                    np.random.randint(0, depth)
                ]),
                'name': f'City_{i}'
            }
        return cities

    @staticmethod
    def create_sphere_cities(num_cities: int, radius: int = 400) -> Dict:
        """
        Create cities distributed on a sphere surface (3D)
        
        Parameters:
        num_cities: Number of cities to generate
        radius: Radius of the sphere
        
        Returns:
        Dictionary of cities on sphere surface
        """
        cities = {}
        for i in range(num_cities):
            # Generate random points on sphere surface
            u = np.random.uniform(0, 1)
            v = np.random.uniform(0, 1)
            
            theta = 2 * np.pi * u  # azimuthal angle
            phi = np.arccos(2 * v - 1)  # polar angle
            
            x = radius * np.sin(phi) * np.cos(theta)
            y = radius * np.sin(phi) * np.sin(theta)
            z = radius * np.cos(phi)
            
            cities[i] = {
                'coord': np.array([x, y, z]),
                'name': f'City_{i}'
            }
        return cities

    @staticmethod
    def load_cities(filename: str) -> Dict:
        """Load cities from pickle file"""
        with open(filename, 'rb') as f:
            return pickle.load(f)
    
    @staticmethod
    def save_cities(cities: Dict, filename: str):
        """Save cities to pickle file"""
        with open(filename, 'wb') as f:
            pickle.dump(cities, f)

# 包含城市信息的字典，格式为:
# {
#     0: {"index": 0, "coord": np.array([x1, y1]), "name": "City 0"},
#     1: {"index": 1, "coord": np.array([x2, y2]), "name": "City 1"},
#     ...
# }

# Example usage and demonstration
# print("TSP Solver Demo - Supporting both 2D and 3D coordinates\n")
#
# # Demo 1: 2D TSP
# print("=" * 50)
# print("Demo 1: 2D TSP Problem")
# print("=" * 50)
#
# cities_2d = TSPSolver.create_random_cities_2d(12, 800, 600)
# solver_2d = TSPSolver(pop_size=100, max_iter=800, mutation_rate=0.1, verbose=True)
# solution_2d = solver_2d.solve(cities_2d)
# solver_2d.plot_solution(solution_2d)
#
# print(f"\n2D Results:")
# print(f"Best route: {' -> '.join(solution_2d['best_route_names'])}")
# print(f"Total distance: {solution_2d['best_distance']:.2f}")
#
# # Demo 2: 3D TSP
# print("\n" + "=" * 50)
# print("Demo 2: 3D TSP Problem")
# print("=" * 50)
#
# cities_3d = TSPSolver.create_random_cities_3d(10, 800, 600, 400)
# solver_3d = TSPSolver(pop_size=100, max_iter=800, mutation_rate=0.1, verbose=True)
# solution_3d = solver_3d.solve(cities_3d)
# solver_3d.plot_solution(solution_3d)
#
# print(f"\n3D Results:")
# print(f"Best route: {' -> '.join(solution_3d['best_route_names'])}")
# print(f"Total distance: {solution_3d['best_distance']:.2f}")
#
# # Demo 3: Cities on sphere surface
# print("\n" + "=" * 50)
# print("Demo 3: TSP on Sphere Surface")
# print("=" * 50)
#
# cities_sphere = TSPSolver.create_sphere_cities(8, 400)
# solver_sphere = TSPSolver(pop_size=100, max_iter=600, mutation_rate=0.1, verbose=True)
# solution_sphere = solver_sphere.solve(cities_sphere)
# solver_sphere.plot_solution(solution_sphere)
#
# print(f"\nSphere Results:")
# print(f"Best route: {' -> '.join(solution_sphere['best_route_names'])}")
# print(f"Total distance: {solution_sphere['best_distance']:.2f}")
#
# # Save examples for later use
# TSPSolver.save_cities(cities_2d, 'cities_2d_example.pkl')
# TSPSolver.save_cities(cities_3d, 'cities_3d_example.pkl')
# TSPSolver.save_cities(cities_sphere, 'cities_sphere_example.pkl')
# print("\nExample city datasets saved to .pkl files!")