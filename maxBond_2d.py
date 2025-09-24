import math
import sys

def main(d: int):
    D = float(d)/2 * (math.sqrt(5)-1)
    return D

if  __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python maxBond_2d.py <int>")
        sys.exit(1)

    d = int(sys.argv[1])  # first argument after the script name
    result = main(d)
    print(result)