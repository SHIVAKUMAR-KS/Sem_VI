jug1, jug2, goal = 4, 3, 2
visited = [[False for _ in range(jug2 + 1)] for _ in range(jug1 + 1)]

def waterJug(vol1, vol2):
    # Check if the solution is found
    if (vol1 == goal and vol2 == 0) or (vol2 == goal and vol1 == 0):
        print(vol1, "\t", vol2)
        print("Solution Found")
        return True

    # If the current state has been visited before, return False
    if visited[vol1][vol2]:
        return False
    
    # Mark the current state as visited
    visited[vol1][vol2] = True
    
    # Print the current state
    print(vol1, "\t", vol2)
    
    # Recursively try different actions for pouring water between jugs
    return (
        waterJug(0, vol2) or  # Empty jug1
        waterJug(vol1, 0) or  # Empty jug2
        waterJug(jug1, vol2) or  # Fill jug1
        waterJug(vol1, jug2) or  # Fill jug2
        waterJug(vol1 + min(vol2, (jug1 - vol1)), vol2 - min(vol2, (jug1 - vol1))) or  # Pour water from jug2 to jug1
        waterJug(vol1 - min(vol1, (jug2 - vol2)), vol2 + min(vol1, (jug2 - vol2)))  # Pour water from jug1 to jug2
    )

# Print header for steps
print("Steps: ")
print("Jug1 \t Jug2")
print("----- \t -----")

# Start the recursion with both jugs empty
waterJug(0, 0)
