import numpy as np
import matplotlib.pyplot as plt

rng = np.random.Generator(np.random.MT19937())

N = 34
EA = 0.66       
EJ = 4.13
omega_a = 1e4   
omega_b = 10.0  
c_ratio = 1.0

def neighbours(i):
    left = (i - 1) % N
    right = (i + 1) % N
    return left, right

def active_neighbours(i):
    left, right = neighbours(i)
    return A[left] + A[right]

def conformational_rate(i):
    n_active = active_neighbours(i)
    delta_G = (EA * (1 - 2 * s[i])+ 2 * EJ * (1 - n_active))
    if A[i] == 0:
        return omega_a * np.exp(-delta_G / 2)
    else:
        return omega_a * np.exp(delta_G / 2)

def cheYP_rate(i):
    if s[i] == 0:
        return omega_b * c_ratio
    elif A[i] == 0:
        return omega_b * np.exp(EA)
    else:
        return omega_b * np.exp(-EA)

def waiting_time(rate):
    u = 1.0 - rng.random()
    return -np.log(u) / rate

# Setting initial rates, waiting times, conformation and binding state
A = np.zeros(N, dtype=int)
s = np.zeros(N, dtype=int)
conformational_rates = np.array([conformational_rate(i) for i in range(N)])
binding_rates = np.array([cheYP_rate(i) for i in range(N)])
T_A = np.array([waiting_time(k) for k in conformational_rates])
T_B = np.array([waiting_time(k) for k in binding_rates])

current_times = [] # Time when an event happened
active_counts = [] # Number of protomers active
bound_counts = [] # Number of protomers bound
event_indices = [] # Which protomer changed 
event_types = []  # Conformational change or binding

current_time = 0.0
t_max = 3

while True:

    min_A = np.min(T_A)
    index_A = np.argmin(T_A)

    min_B = np.min(T_B)
    index_B = np.argmin(T_B)

    if min_A <= min_B: # Event type and protomer number record 
        event_type = "Conformational"
        i = index_A
        next_time = min_A
    else:
        event_type = "Binding/Unbinding"
        i = index_B
        next_time = min_B

    if next_time > t_max:
        break

    current_time = next_time # Current time record

    # Update the rate and next waiting times for ith protomer's "neighbours"(i+1,i-1)
    if event_type == "Conformational":
        A[i] = 1 - A[i]
        left, right = neighbours(i)
        conformational_rates[left] = (conformational_rate(left))
        T_A[left] = (current_time+ waiting_time(conformational_rates[left]))
        conformational_rates[right] = (conformational_rate(right))
        T_A[right] = (current_time+ waiting_time(conformational_rates[right]))
    else:
        s[i] = 1 - s[i]

    # Update the rate and next waiting times for ith protomer
    conformational_rates[i] = (conformational_rate(i))
    binding_rates[i] = cheYP_rate(i)
    T_A[i] = (current_time+ waiting_time(conformational_rates[i]))
    T_B[i] = (current_time+ waiting_time(binding_rates[i]))

    lists = [current_times, active_counts, bound_counts, event_indices, event_types]
    values = [current_time, np.sum(A), np.sum(s), i, event_type]
    for j, value in zip(lists, values):
        j.append(value)


plt.figure(figsize=(10, 5))
plt.plot(current_times, active_counts, color="blue", label="Active Protomers")
plt.plot(current_times, bound_counts, color="red", label="Bound CheY-P")
plt.xlabel("Time")
plt.ylabel("Number of Protomers(Active/Bound)")
plt.title("Symmetric Conformational Spread Model")
plt.ylim(0, 35)
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()


for j in lists:
    print(len(j))
print(np.mean(active_counts))
print(np.mean(bound_counts))
