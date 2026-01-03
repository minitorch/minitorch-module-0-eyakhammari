"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


def mul(x: float, y: float) -> float:
    """Multiply two numbers.

    Args:
    ----
        x: First number
        y: Second number

    Returns:
    -------
        Product of x and y

    """
    return x * y


def id(x: float) -> float:
    """Return the input unchanged (identity function).

    Args:
    ----
        x: Input number

    Returns:
    -------
        The same number x

    """
    return x


def add(x: float, y: float) -> float:
    """Add two numbers.

    Args:
    ----
        x: First number
        y: Second number

    Returns:
    -------
        Sum of x and y

    """
    return x + y


def neg(x: float) -> float:
    """Negate a number.

    Args:
    ----
        x: Input number

    Returns:
    -------
        Negation of x

    """
    return -x


def lt(x: float, y: float) -> float:
    """Check if x is less than y.

    Args:
    ----
        x: First number
        y: Second number

    Returns:
    -------
        1.0 if x < y, otherwise 0.0

    """
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    """Check if x equals y.

    Args:
    ----
        x: First number
        y: Second number

    Returns:
    -------
        1.0 if x == y, otherwise 0.0

    """
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """Return the maximum of two numbers.

    Args:
    ----
        x: First number
        y: Second number

    Returns:
    -------
        The larger of x and y

    """
    return x if x > y else y


def is_close(x: float, y: float) -> float:
    """Check if two numbers are close (within 1e-2).

    Args:
    ----
        x: First number
        y: Second number

    Returns:
    -------
        1.0 if |x - y| < 1e-2, otherwise 0.0

    """
    return 1.0 if abs(x - y) < 1e-2 else 0.0


def sigmoid(x: float) -> float:
    """Calculate sigmoid function.

    For numerical stability:
    - If x >= 0: f(x) = 1.0 / (1.0 + e^(-x))
    - If x < 0: f(x) = e^x / (1.0 + e^x)

    Args:
    ----
        x: Input number

    Returns:
    -------
        Sigmoid of x

    """
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Apply ReLU activation function.

    Args:
    ----
        x: Input number

    Returns:
    -------
        x if x > 0, otherwise 0.0

    """
    return x if x > 0 else 0.0


def log(x: float) -> float:
    """Calculate natural logarithm.

    Args:
    ----
        x: Input number

    Returns:
    -------
        Natural log of x

    """
    return math.log(x)


def exp(x: float) -> float:
    """Calculate exponential function.

    Args:
    ----
        x: Input number

    Returns:
    -------
        e^x

    """
    return math.exp(x)


def inv(x: float) -> float:
    """Calculate reciprocal (inverse).

    Args:
    ----
        x: Input number

    Returns:
    -------
        1/x

    """
    return 1.0 / x


def log_back(x: float, d: float) -> float:
    """Calculate derivative of log times a derivative.

    Args:
    ----
        x: Input number
        d: Derivative from subsequent computation

    Returns:
    -------
        d * (1/x)

    """
    return d / x


def inv_back(x: float, d: float) -> float:
    """Calculate derivative of reciprocal times a derivative.

    Args:
    ----
        x: Input number
        d: Derivative from subsequent computation

    Returns:
    -------
        d * (-1/x^2)

    """
    return -d / (x * x)


def relu_back(x: float, d: float) -> float:
    """Calculate derivative of ReLU times a derivative.

    Args:
    ----
        x: Input number
        d: Derivative from subsequent computation

    Returns:
    -------
        d if x > 0, otherwise 0.0

    """
    return d if x > 0 else 0.0


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], list[float]]:
    """Higher-order map function.

    Args:
    ----
        fn: Function to apply to each element

    Returns:
    -------
        Function that maps fn over an iterable

    """
    def _map(ls: Iterable[float]) -> list[float]:
        return [fn(x) for x in ls]

    return _map


def zipWith(
    fn: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], list[float]]:
    """Higher-order zipWith function.

    Args:
    ----
        fn: Function to combine elements

    Returns:
    -------
        Function that combines two iterables element-wise

    """
    def _zipWith(ls1: Iterable[float], ls2: Iterable[float]) -> list[float]:
        return [fn(x, y) for x, y in zip(ls1, ls2)]

    return _zipWith


def reduce(
    fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    """Higher-order reduce function.

    Args:
    ----
        fn: Function to combine elements
        start: Starting value for reduction

    Returns:
    -------
        Function that reduces an iterable to a single value

    """
    def _reduce(ls: Iterable[float]) -> float:
        result = start
        for x in ls:
            result = fn(result, x)
        return result

    return _reduce


def negList(ls: Iterable[float]) -> list[float]:
    """Negate all elements in a list.

    Args:
    ----
        ls: Input list

    Returns:
    -------
        List with all elements negated

    """
    return map(neg)(ls)


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> list[float]:
    """Add corresponding elements from two lists.

    Args:
    ----
        ls1: First list
        ls2: Second list

    Returns:
    -------
        List of sums

    """
    return zipWith(add)(ls1, ls2)


def sum(ls: Iterable[float]) -> float:
    """Sum all elements in a list.

    Args:
    ----
        ls: Input list

    Returns:
    -------
        Sum of all elements

    """
    return reduce(add, 0.0)(ls)


def prod(ls: Iterable[float]) -> float:
    """Calculate product of all elements in a list.

    Args:
    ----
        ls: Input list

    Returns:
    -------
        Product of all elements

    """
    return reduce(mul, 1.0)(ls)
