# Strategic Network Optimization: A Data-Driven Approach to the Multi-Facility Location Problem

This project was developed as part of the **IE 440: Nonlinear Models** course at **Boğaziçi University** by me and my friends Mete Kalkan and Yiğit Bostancı.

It applies nonlinear optimization techniques to solve the classic Multi-Facility Weber Problem: determining the optimal locations for `m` facilities (e.g., distribution centers) to serve `n` customers (e.g., retail stores) to minimize total weighted transportation costs.

This analytical challenge is at the core of strategic business decisions, directly addressing questions like:
> "Where should a retailer open new stores to maximize profitability?"

This repository contains the full Python implementation (using Pandas, Matplotlib, and Scipy) from data ingestion to final strategic recommendations.

### 📈 Final Recommendation: Optimized Network Map





---

## The Business & Data Context

The objective is to design the lowest-cost logistics network for a hypothetical client.

* **Network Size:** **100** demand points (customers/stores).
* **Infrastructure:** **50** potential supply points (facilities/distribution centers).
* **The Data:** The model uses three key inputs from the `/data` folder:
    1.  `coordinates_9.csv`: Geographic (x, y) coordinates for all 100 customers.
    2.  `demands_9.csv`: The demand volume ($h_j$) for each customer.
    3.  `costs_9.csv`: A full $50 \times 100$ cost matrix ($c_{ij}$), representing the unique cost-per-distance of serving customer $j$ from facility $i$.
* **Objective Function:** Minimize the total weighted distance cost:
    $$Z = \sum_{i=1}^{50} \sum_{j=1}^{100} y_{ij} \cdot (h_j \cdot c_{ij}) \cdot d(x_i, a_j)$$

---

## Methodology: A Four-Phase Analytical Approach

The problem was solved in a logical progression, starting with a simple baseline and building up to a robust, multi-facility simulation.

### Phase 1: Baseline Model (Single Facility, Squared Distance)
To establish a simple baseline, we first solved the problem for a single central facility assuming costs are proportional to the **Squared Euclidean distance**.
* **Method:** This model has a direct analytical "center of gravity" solution.
* **Result:** The optimal location was found to be **[20.34, 19.65]** with a total cost of 373,834.
* **Limitation:** This is a poor proxy for real-world logistics, where cost is linear to distance, not its square.

### Phase 2: Realistic Model (Single Facility, Euclidean Distance)
Next, we solved for a single facility using the **true Euclidean distance**. This is a nonlinear optimization problem that requires an iterative solver.
* **Method:** Implemented **Weiszfeld's Algorithm** from scratch in Python. This algorithm iteratively computes a new weighted average location until the solution converges.
* **Result:** The algorithm converged at **[20.59, 19.39]**, a demonstrably different and more accurate location than the simplistic baseline model.

### Phase 3: The Multi-Facility Solution (Business Experimentation)
This is the core of the project, solving the full problem of locating 50 facilities.
* **Method:** Implemented the **Alternate Location-Allocation (ALA) Heuristic**. This is a powerful iterative algorithm that mimics real-world strategic planning:
    1.  **Allocation Step:** With fixed facility locations, assign each customer to the facility that offers the lowest weighted cost.
    2.  **Location Step:** With the new customer assignments (clusters), re-calculate the optimal location for *each* facility independently using the Weiszfeld's Algorithm from Phase 2.
* **Process:** These two steps are repeated until the network converges and the total cost ($Z$) no longer improves.

### Phase 4: Robustness Testing (Performance Analytics)
A heuristic's solution can be sensitive to its starting guess. A good analyst must test for this.
* **Method:** To find a truly robust solution and avoid a "lucky" local minimum, I ran the entire ALA heuristic **1,000 times**, each with a different random initial assignment of customers to facilities.
* **Goal:** This "simulation-based approach" allows me to run a **data-backed business experiment** to find the *best possible* network configuration and understand its performance against the average.

---

## 💡 Key Findings & Actionable Insights

The analysis of the 1,000-trial simulation (using the realistic Euclidean model) provided clear, data-driven recommendations.

### 1. The Value of Optimization
* **Best-Case Scenario (The Recommendation):** The best network configuration found across all 1,000 trials achieved a minimum total cost of **1,310.99**.
* **Average-Case Scenario (The Baseline):** The *average* cost across all simulations was **1,811.20**.

> **Actionable Insight:** The optimized solution is **27.6% more cost-efficient** than the average randomly-generated strategy. This figure quantifies the direct financial value of applying this analytical model versus a purely intuitive or random approach.

### 2. Strategic Network Consolidation
* **Finding:** In the optimal solution, several potential facilities (e.g., Facility 27) were assigned **zero customers**.
* **Reason:** The algorithm found that it was more cost-effective for other nearby facilities to serve that area.

> **Strategic Recommendation:** The client does not need to open all 50 potential facilities. This analysis identifies which locations are redundant, allowing for a **network consolidation strategy**. This would lead to significant **capital expenditure (CapEx) savings** (from not building unused facilities) on top of the **operational expenditure (OpEx) savings** from optimized transport.

---

## 🛠️ Technical Stack

* **Core Language:** Python
* **Data Analysis & Handling:** Pandas, NumPy
* **Data Visualization:** Matplotlib
* **Core Algorithms:**
    * Weiszfeld's Algorithm (Nonlinear Iterative Solver)
    * Alternate Location-Allocation (ALA) Heuristic
    * Monte Carlo Simulation (1,000 trials) for Heuristic Optimization
