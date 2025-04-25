from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import numpy as np
import json
import time

app = Flask(__name__)
CORS(app)

def objective_function(solution):
    A = 10
    return A*len(solution) + np.sum(solution**2 - A*np.cos(2*np.pi*solution))

def artificial_bee_colony(max_iterations, colony_size, dimensions):
    
    population = np.random.uniform(-5.12, 5.12, (colony_size, dimensions))
    fitness = np.array([objective_function(sol) for sol in population])
    best_solution = population[np.argmin(fitness)]
    best_fitness = np.min(fitness)

    for iteration in range(max_iterations):
      
        for i in range(colony_size):
            k = np.random.randint(0, colony_size)
            while k == i:
                k = np.random.randint(0, colony_size)
            phi = np.random.uniform(-1.5, 1.5, dimensions)  
            new_solution = population[i] + phi * (population[i] - population[k])
            
            new_solution = np.clip(new_solution, -5.12, 5.12)
            new_fitness = objective_function(new_solution)

            if new_fitness < fitness[i]:
                population[i] = new_solution
                fitness[i] = new_fitness

        
        fitness_prob = 1 / (1 + fitness) 
        fitness_prob /= np.sum(fitness_prob)  

        for i in range(colony_size):
            if np.random.rand() < fitness_prob[i]:  
                k = np.random.randint(0, colony_size)
                while k == i:
                    k = np.random.randint(0, colony_size)
                phi = np.random.uniform(-1.5, 1.5, dimensions)
                new_solution = population[i] + phi * (population[i] - population[k])
                new_solution = np.clip(new_solution, -5.12, 5.12)
                new_fitness = objective_function(new_solution)

                if new_fitness < fitness[i]:
                    population[i] = new_solution
                    fitness[i] = new_fitness

        
        for i in range(colony_size):
            if fitness[i] > np.percentile(fitness, 90):  
                population[i] = np.random.uniform(-5, 5, dimensions)
                fitness[i] = objective_function(population[i])

        
        current_best_fitness = np.min(fitness)
        if current_best_fitness < best_fitness:
            best_solution = population[np.argmin(fitness)]
            best_fitness = current_best_fitness

        yield iteration, best_solution, best_fitness, np.mean(fitness)


@app.route('/abc-optimize-stream', methods=['GET'])
def abc_optimize_stream():
    max_iterations = request.args.get('max_iterations', default=100, type=int)
    colony_size = request.args.get('colony_size', default=20, type=int)
    dimensions = request.args.get('dimensions', default=2, type=int)
    
    start_time = time.time() 

    def generate():
        abc_gen = artificial_bee_colony(max_iterations, colony_size, dimensions)
        for iteration, best_solution, best_fitness, mean_fitness in abc_gen:
            elapsed_time = time.time() - start_time  
            data = {
                'iteration': iteration + 1,
                'best_solution': best_solution.tolist(),
                'best_fitness': float(best_fitness),
                'mean_fitness': float(mean_fitness),
                'progress': ((iteration + 1) / max_iterations) * 100,
                'computation_time': elapsed_time  
            }
            yield f"data: {json.dumps(data)}\n\n"

    return Response(generate(), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(debug=True)
