import time 

def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        results = func(*args, **kwargs)
        duration = time.time() - start
        print("Function '{}' finished after {:.4f} seconds."\
              .format(func.__name__, duration))
        return results
    return wrapper