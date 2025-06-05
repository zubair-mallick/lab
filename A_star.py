
import heapq

GRID_SIZE = 10
start = (0, 0)
goal = (9, 9)

# 0 = free, 1 = obstacle
grid = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar(start, goal):
    heap = [(0, start)]
    came_from = {}
    cost = {start: 0}

    while heap:
        _, current = heapq.heappop(heap)

        if current == goal:
            path = []
            while current != start:
                path.append(current)
                current = came_from[current]
            return [start] + path[::-1]

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            x, y = current[0] + dx, current[1] + dy
            if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE and grid[x][y] == 0:
                next = (x, y)
                new_cost = cost[current] + 1
                if next not in cost or new_cost < cost[next]:
                    cost[next] = new_cost
                    heapq.heappush(heap, (new_cost + heuristic(next, goal), next))
                    came_from[next] = current

    return None

# Run it
path = astar(start, goal)
print("✅ Path:" if path else "❌ No path found", path)
