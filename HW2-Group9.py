import csv
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def plot_single_facility(customers_x, customers_y, facility_location, title):
    plt.figure(figsize=(8, 6))
    plt.scatter(customers_x, customers_y, color='gray', alpha=0.6, label='Customers')
    plt.scatter(facility_location[0], facility_location[1], color='red', marker='X', s=200, label='Facility')
    plt.text(facility_location[0], facility_location[1], 'Facility', color='black', fontsize=10,
             fontweight='bold', ha='left', va='bottom')
    plt.xlabel('X1 Coordinate')
    plt.ylabel('X2 Coordinate')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_facilities(customers_x, customers_y, facility_locations, facility_assignments):
    plt.figure(figsize=(12, 9))

    num_facilities = len(facility_locations)
    colors = cm.get_cmap('tab20', num_facilities)

    # Plot each customer's location with the color of their facility
    for facility_index, customer_indices in facility_assignments.items():
        if not customer_indices:
            continue
        x_coords = [customers_x[i] for i in customer_indices]
        y_coords = [customers_y[i] for i in customer_indices]
        plt.scatter(x_coords, y_coords, color=colors(facility_index), alpha=0.6)

    # Plot facilities with locations
    for facility_index, loc in facility_locations.items():
        if loc is not None:
            plt.scatter(loc[0], loc[1], color=colors(facility_index), marker='X', s=200)
            plt.text(loc[0], loc[1], str(facility_index), color='black', fontsize=9,
                     fontweight='bold', ha='center', va='center')


    plt.xlabel('X1 Coordinate')
    plt.ylabel('X2 Coordinate')
    plt.title('Customer Assignments to Facilities')
    plt.grid(True)
    plt.show()
    
random.seed(0)

x1_coordinates = []
x2_coordinates = []

with open(r'coordinates_9.csv', 'r', newline='') as f: 
    for row in csv.reader(f):
        coordinates = list(float(item) for item in row)
        x1_coordinates.append(coordinates[0])
        x2_coordinates.append(coordinates[1])

costs = []

with open(r'costs_9.csv', 'r', newline='') as f: 
    for row in csv.reader(f):
        costs.append(list(float(item) for item in row)) # Note: costs is a list of lists

demands = []

with open(r'demand_9.csv', 'r', newline='') as f: 
    for row in csv.reader(f):
        demands.append(float(row[0]))

def squared_euclidian_distance_single_facility(cost_list, demand_list, coordinates_x1, coordinates_x2):
    numerator = 0
    denominator = 0
    n = len(demand_list) # Use the actual length of the data
    
    for j in range(n): # Use loop variable j and iterate up to n
        numerator += demand_list[j] * cost_list[j] * coordinates_x1[j]
        denominator += demand_list[j] * cost_list[j]
    
    x1 = numerator / denominator
    numerator = 0 # Reset numerator for y-coordinate calculation
    # The denominator will have the same value for the x2 coordinate, no need to reset or calculate again
    
    for j in range(n):
        numerator += demand_list[j] * cost_list[j] * coordinates_x2[j]
    
    x2 = numerator / denominator
    x_star = [x1, x2] # Final coordinate
    f_star = 0
    
    for j in range(n):
        f_star += demand_list[j] * cost_list[j] * ((x_star[0] - coordinates_x1[j])**2 + (x_star[1] - coordinates_x2[j])**2)
        # Calculating the final cost value
    
    return x_star, f_star 

selected_cost = costs[0] # selecting facility 1 for the single facility minimization

loc1, val1 = squared_euclidian_distance_single_facility(selected_cost, demands, x1_coordinates, x2_coordinates)
print("Final location for the single facility (Squared Euclidian):", loc1)
print("Total cost of the single facility (Squared Euclidian):", val1)
plot_single_facility(x1_coordinates, x2_coordinates, loc1, "Single Facility (Squared Euclidean)")

def weiszfelds_algorithm_euclidean_distance_single_facility(cost_list, demand_list, coordinates_x1, coordinates_x2):
    epsilon = 0.000001
    x_k = squared_euclidian_distance_single_facility(cost_list, demand_list, coordinates_x1, coordinates_x2)[0]
    x_k_plus_1 = []
    n = len(demand_list)
    
    while True:
        numerator_x1 = 0
        numerator_x2 = 0
        denominator = 0

        for i in range(n):
            distance = ((x_k[0] - coordinates_x1[i])**2 + (x_k[1] - coordinates_x2[i])**2)**0.5
            if distance < epsilon: # handle  the case where distance is very close to zero to avoid division by zero
                distance = epsilon
            weight = demand_list[i] * cost_list[i] / distance
            numerator_x1 += weight * coordinates_x1[i]
            numerator_x2 += weight * coordinates_x2[i]
            denominator += weight

        x_k_plus_1 = [numerator_x1 / denominator, numerator_x2 / denominator]

        if ((x_k_plus_1[0] - x_k[0])**2 + (x_k_plus_1[1] - x_k[1])**2)**0.5 < epsilon: # stopping condition
            break
        
        x_k = x_k_plus_1[:]

    x_star = x_k_plus_1
    f_star = 0

    for j in range(n):
        distance = ((x_star[0] - coordinates_x1[j])**2 + (x_star[1] - coordinates_x2[j])**2)**0.5
        f_star += demand_list[j] * cost_list[j] * distance

    return x_star, f_star

loc2, val2 = weiszfelds_algorithm_euclidean_distance_single_facility(selected_cost, demands, x1_coordinates, x2_coordinates)
print("Final location for the single facility (Weiszfeld’s Algorithm):", loc2)
print("Total cost of the single facility (Weiszfeld’s Algorithm):", val2)
plot_single_facility(x1_coordinates, x2_coordinates, loc2, "Single Facility (Weiszfeld’s Algorithm)")

def squared_euclidian_distance_multi_facility(cost_list, demand_list, coordinates_x1, coordinates_x2):
    # Define the number of facilities
    num_facilities = len(cost_list)

    # Define the number of customers
    num_customers = len(demand_list)

    # Create a list to store the facility assignment for each customer
    # Ensure each facility has at least one customer initially, then randomly assign the rest
    customer_facility_assignment = list(range(num_facilities)) # Assign the first num_facilities customers uniquely
    remaining_assignments = [random.randint(0, num_facilities - 1) for _ in range(num_customers - num_facilities)]
    customer_facility_assignment.extend(remaining_assignments)
    random.shuffle(customer_facility_assignment) # Shuffle to maintain randomness
    
    facility_customer_assignment = {i: [] for i in range(num_facilities)}
    for i in range(num_customers):
        facility = customer_facility_assignment[i]
        facility_customer_assignment[facility].append(i)
    
    objective_function_value_k = float('inf') # Initialize with a infinitely large value
    objective_function_value_k_plus_1 = 1000000000000 # Initialize also with a large value, but less than infinite so that the while loop can begin
    facility_locations = {}


    # Iterative process
    while objective_function_value_k - objective_function_value_k_plus_1 > 0: # Convergence criterion
        objective_function_value_k = objective_function_value_k_plus_1 # Update current objective value
        objective_function_value_k_plus_1 = 0 # Reset next objective value

        # Initialize dictionaries to store grouped data
        grouped_demand = {i: [] for i in range(num_facilities)}
        grouped_cost = {i: [] for i in range(num_facilities)}
        grouped_coordinates_x1 = {i: [] for i in range(num_facilities)}
        grouped_coordinates_x2 = {i: [] for i in range(num_facilities)}

        # Group the data based on customer_facility_assignment
        for i in range(num_customers):
            facility_index = customer_facility_assignment[i]
            grouped_demand[facility_index].append(demand_list[i])
            grouped_cost[facility_index].append(cost_list[facility_index][i])
            grouped_coordinates_x1[facility_index].append(coordinates_x1[i])
            grouped_coordinates_x2[facility_index].append(coordinates_x2[i])

        # Calculate facility locations using squared_euclidian_distance_single_facility for each group
        for facility_index in range(num_facilities):
            if grouped_demand[facility_index]:
                facility_location = squared_euclidian_distance_single_facility(
                grouped_cost[facility_index],
                grouped_demand[facility_index],
                grouped_coordinates_x1[facility_index],
                grouped_coordinates_x2[facility_index]
            )[0]
                facility_locations[facility_index] = facility_location
            
        # Calculate new objective value
        for facility in facility_customer_assignment:
            customers = facility_customer_assignment[facility]
            if customers:
                for customer in customers:
                    customer_location = [coordinates_x1[customer], coordinates_x2[customer]]
                    customer_demand = demand_list[customer]
                    customer_cost = cost_list[facility][customer]
                    if facility_locations[facility] is not None:
                        distance_squared = (facility_locations[facility][0] - customer_location[0])**2 + (facility_locations[facility][1] - customer_location[1])**2
                        objective_function_value_k_plus_1 += customer_demand * customer_cost * distance_squared

        # Reassign customers to nearest facilities
        new_customer_facility_assignment = []
        new_facility_customer_assignment = {i: [] for i in range(num_facilities)}

        for i in range(num_customers):
            customer_location = [coordinates_x1[i], coordinates_x2[i]]
            min_weighted_distance_squared = float('inf')
            nearest_facility_index = -1

            for facility_index, facility_location in facility_locations.items():
                if facility_location is not None: # Only consider facilities with a calculated location
                    # Calculate squared Euclidean distance
                    distance_squared = (customer_location[0] - facility_location[0])**2 + (customer_location[1] - facility_location[1])**2

                    # Calculate weighted distance squared
                    weighted_distance_squared = demand_list[i] * cost_list[facility_index][i] * distance_squared

                    if weighted_distance_squared < min_weighted_distance_squared:
                        min_weighted_distance_squared = weighted_distance_squared
                        nearest_facility_index = facility_index

            new_customer_facility_assignment.append(nearest_facility_index)
           
        for i in range(num_customers):
            facility = new_customer_facility_assignment[i]
            new_facility_customer_assignment[facility].append(i)

        customer_facility_assignment = new_customer_facility_assignment 
        facility_customer_assignment = new_facility_customer_assignment # Update assignments for the next iteration

    return facility_locations, facility_customer_assignment, objective_function_value_k_plus_1, num_facilities

# Run the multi-facility algorithm with 1000 different seeds to find best heuristic result
best_result = float('inf')
best_seed = 0
average1 = 0
for i in range(1000):
    random.seed(i)
    current_result = squared_euclidian_distance_multi_facility(costs, demands, x1_coordinates, x2_coordinates)[2]
    average1 += current_result / 1000
    if current_result < best_result:
        best_result = current_result
        best_seed = i

random.seed(best_seed)

final_facility_locations, final_facility_assignments, final_objective_value, num_facilities = \
    squared_euclidian_distance_multi_facility(costs, demands, x1_coordinates, x2_coordinates)

def print_out(final_facility_locations, final_facility_assignments, final_objective_value, num_facilities):

    print("\nFinal Facility Locations:")
    for facility_index, location in final_facility_locations.items():
        print(f"Facility {facility_index}: {location}")

    # Section to display final customer assignments by facility
    print("\nFinal Customer Assignments:")

    # Print the customers for each facility
    for facility_index in range(num_facilities):
        print(f"Facility {facility_index}: Customers {final_facility_assignments[facility_index]}")

    # Print the objective value
    print(f"\nFinal Objective Function Value: {final_objective_value}")

print_out(final_facility_locations, final_facility_assignments, final_objective_value, num_facilities)
print("Average of 1000 trials:", average1)


plot_facilities(x1_coordinates, x2_coordinates, final_facility_locations, final_facility_assignments)

def euclidian_distance_multi_facility(cost_list, demand_list, coordinates_x1, coordinates_x2):
    # Define the number of facilities
    num_facilities = len(cost_list)

    # Define the number of customers
    num_customers = len(demand_list)

    # Create a list to store the facility assignment for each customer
    # Ensure each facility has at least one customer initially, then randomly assign the rest
    customer_facility_assignment = list(range(num_facilities)) # Assign the first num_facilities customers uniquely
    remaining_assignments = [random.randint(0, num_facilities - 1) for _ in range(num_customers - num_facilities)]
    customer_facility_assignment.extend(remaining_assignments)
    random.shuffle(customer_facility_assignment) # Shuffle to maintain randomness
    
    facility_customer_assignment = {i: [] for i in range(num_facilities)}
    for i in range(num_customers):
        facility = customer_facility_assignment[i]
        facility_customer_assignment[facility].append(i)
    
    objective_function_value_k = float('inf') # Initialize with a large value
    objective_function_value_k_plus_1 = 1000000000000
    facility_locations = {}

    # Iterative process
    while objective_function_value_k - objective_function_value_k_plus_1 > 0: # Convergence criterion
        objective_function_value_k = objective_function_value_k_plus_1 # Update current objective value
        objective_function_value_k_plus_1 = 0 # Reset next objective value

        # Initialize dictionaries to store grouped data
        grouped_demand = {i: [] for i in range(num_facilities)}
        grouped_cost = {i: [] for i in range(num_facilities)}
        grouped_coordinates_x1 = {i: [] for i in range(num_facilities)}
        grouped_coordinates_x2 = {i: [] for i in range(num_facilities)}

        # Group the data based on customer_facility_assignment
        for i in range(num_customers):
            facility_index = customer_facility_assignment[i]
            grouped_demand[facility_index].append(demand_list[i])
            grouped_cost[facility_index].append(cost_list[facility_index][i])
            grouped_coordinates_x1[facility_index].append(coordinates_x1[i])
            grouped_coordinates_x2[facility_index].append(coordinates_x2[i])

        # Calculate facility locations using squared_euclidian_distance_single_facility for each group
        for facility_index in range(num_facilities):
            if grouped_demand[facility_index]:
                facility_location = weiszfelds_algorithm_euclidean_distance_single_facility(grouped_cost[facility_index],
                grouped_demand[facility_index],
                grouped_coordinates_x1[facility_index],
                grouped_coordinates_x2[facility_index]
                )[0]
                
                facility_locations[facility_index] = facility_location
            

        # Calculate new objective value
        for facility in facility_customer_assignment:
            customers = facility_customer_assignment[facility]
            if customers:
                for customer in customers:
                    customer_location = [coordinates_x1[customer], coordinates_x2[customer]]
                    customer_demand = demand_list[customer]
                    customer_cost = cost_list[facility][customer]
                    if facility_locations[facility] is not None:
                        distance = ((facility_locations[facility][0] - customer_location[0])**2 + (facility_locations[facility][1] - customer_location[1])**2)**0.5
                        objective_function_value_k_plus_1 += customer_demand * customer_cost * distance

        # Reassign customers to nearest facilities
        new_customer_facility_assignment = []
        new_facility_customer_assignment = {i: [] for i in range(num_facilities)}

        for i in range(num_customers):
            customer_location = [coordinates_x1[i], coordinates_x2[i]]
            min_weighted_distance = float('inf')
            nearest_facility_index = -1

            for facility_index, facility_location in facility_locations.items():
                if facility_location is not None: # Only consider facilities with a calculated location
                    # Calculate Euclidean distance
                    distance = ((customer_location[0] - facility_location[0])**2 + (customer_location[1] - facility_location[1])**2)**0.5

                    # Calculate weighted distance
                    weighted_distance = demand_list[i] * cost_list[facility_index][i] * distance

                    if weighted_distance < min_weighted_distance:
                        min_weighted_distance = weighted_distance
                        nearest_facility_index = facility_index

            new_customer_facility_assignment.append(nearest_facility_index)
           
        for i in range(num_customers):
            facility = new_customer_facility_assignment[i]
            new_facility_customer_assignment[facility].append(i)

        customer_facility_assignment = new_customer_facility_assignment 
        facility_customer_assignment = new_facility_customer_assignment # Update assignments for the next iteration

    return facility_locations, facility_customer_assignment, objective_function_value_k_plus_1, num_facilities

# Run the multi-facility algorithm with 1000 different seeds to find best heuristic result
best_result = float('inf')
best_seed = 0
average2 = 0
for i in range(1000):
    random.seed(i)
    current_result = euclidian_distance_multi_facility(costs, demands, x1_coordinates, x2_coordinates)[2]
    average2 += current_result / 1000
    if current_result < best_result:
        best_result = current_result
        best_seed = i

random.seed(best_seed)

final_facility_locations, final_facility_assignments, final_objective_value, num_facilities = \
    euclidian_distance_multi_facility(costs, demands, x1_coordinates, x2_coordinates)
print("Best seed:", best_seed)
print_out(final_facility_locations, final_facility_assignments, final_objective_value, num_facilities)
print("Average of 1000 trials:", average2)

plot_facilities(x1_coordinates, x2_coordinates, final_facility_locations, final_facility_assignments)