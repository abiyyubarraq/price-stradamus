# Module 01: Python for TypeScript Developers

**Duration:** 2-3 hours | **Difficulty:** Beginner | **Prerequisites:** TypeScript/JavaScript experience

## 🎯 Learning Objectives

After this module, you will:
- Read and write Python code with confidence
- Understand Python syntax compared to TypeScript
- Use type hints (Python's TypeScript equivalent)
- Work with classes, functions, and decorators
- Understand Python's key differences from JavaScript

---

## 📋 Table of Contents

1. [The Big Picture](#the-big-picture)
2. [Syntax Comparison](#syntax-comparison)
3. [Variables and Types](#variables-and-types)
4. [Functions](#functions)
5. [Classes and Objects](#classes-and-objects)
6. [Type Hints (TypeScript-like)](#type-hints-typescript-like)
7. [Collections and Iteration](#collections-and-iteration)
8. [Comprehensions](#comprehensions)
9. [Decorators](#decorators)
10. [Modules and Imports](#modules-and-imports)
11. [Error Handling](#error-handling)
12. [Quick Reference](#quick-reference)
13. [Practice Exercises](#practice-exercises)

---

## The Big Picture

### Mental Model

**TypeScript:** "JavaScript with types"
**Python:** "Readable code that happens to have optional types"

Both languages:
- Are interpreted (no manual compilation step like C++)
- Support dynamic typing (but encourage static types)
- Have first-class functions
- Support OOP and functional programming

### Key Philosophy Differences

| Aspect | TypeScript | Python |
|--------|-----------|--------|
| **Syntax** | C-style (braces, semicolons) | Indentation-based (no braces) |
| **Types** | Required with `strict` mode | Optional with type hints |
| **Philosophy** | "Scale JavaScript" | "Readable is better than clever" |
| **Async** | Promises everywhere | Native async/await (like Node.js) |
| **null handling** | `undefined` and `null` | `None` |

---

## Syntax Comparison

### Basic Syntax

#### TypeScript
```typescript
// TypeScript
function greet(name: string): string {
  return `Hello, ${name}!`;
}

const result = greet("World");
console.log(result);
```

#### Python
```python
# Python
def greet(name: str) -> str:
    return f"Hello, {name}!"

result = greet("World")
print(result)
```

### Key Observations

1. **No braces `{}`** - Python uses indentation (4 spaces is standard)
2. **No semicolons** - Line breaks are statement separators
3. **Colon `:`** - Starts code blocks (functions, classes, if statements)
4. **Comments** - `#` instead of `//`
5. **String interpolation** - `f"..."` instead of template literals

---

## Variables and Types

### Variable Declaration

#### TypeScript
```typescript
// TypeScript
let count: number = 0;
const name: string = "Bitcoin";
var oldStyle: number = 1; // Avoid

// Type inference
let price = 50000; // Inferred as number
```

#### Python
```python
# Python
count: int = 0
name: str = "Bitcoin"
# No 'var' - just use the variable

# Type inference (type hint optional)
price = 50000  # Python knows it's an int
```

**Key Difference:** Python has no `let`, `const`, or `var`. Variables are just declared by assignment. Use type hints for clarity.

### Basic Types

| TypeScript | Python | Example |
|-----------|--------|---------|
| `number` | `int` or `float` | `42` or `3.14` |
| `string` | `str` | `"hello"` |
| `boolean` | `bool` | `True` / `False` (capitalized!) |
| `null` / `undefined` | `None` | `None` |
| `any` | `Any` (from typing) | Use sparingly |

#### Example
```python
# Python types
age: int = 25
price: float = 99.99
name: str = "Alice"
is_active: bool = True
nothing: None = None
```

### None vs null/undefined

**TypeScript:**
```typescript
let value: string | null = null;
let missing: string | undefined = undefined;
```

**Python:**
```python
value: str | None = None
# Only None (no separate undefined)
```

**Example in Price Stradamus:**
```python
# From src/price_stradamus/models/base.py
def load(self, path: Path | None = None) -> None:
    """Load model from disk.

    Args:
        path: Optional path to model file
    """
    if path is None:
        path = self.get_default_path()
```

---

## Functions

### Basic Functions

#### TypeScript
```typescript
// TypeScript
function add(a: number, b: number): number {
  return a + b;
}

const multiply = (a: number, b: number): number => {
  return a * b;
};
```

#### Python
```python
# Python
def add(a: int, b: int) -> int:
    return a + b

# No arrow functions, but lambda for simple cases
multiply = lambda a, b: a * b
```

### Default Parameters

#### TypeScript
```typescript
function greet(name: string = "Guest"): string {
  return `Hello, ${name}`;
}
```

#### Python
```python
def greet(name: str = "Guest") -> str:
    return f"Hello, {name}"
```

### Named Arguments (Python-specific)

Python has a powerful feature TypeScript doesn't have:

```python
def create_user(name: str, age: int, email: str, active: bool = True):
    return {"name": name, "age": age, "email": email, "active": active}

# Call with named arguments (order doesn't matter!)
user = create_user(
    age=25,
    name="Alice",
    email="alice@example.com"
)
```

**Why this matters:** You'll see this everywhere in Price Stradamus:
```python
# From training a model
model = NBEATSModel(
    input_chunk_length=60,
    output_chunk_length=5,
    num_stacks=30,
    n_epochs=100,
)
```

### Multiple Return Values

#### TypeScript
```typescript
// TypeScript - need tuple or object
function divmod(a: number, b: number): [number, number] {
  return [Math.floor(a / b), a % b];
}

const [quotient, remainder] = divmod(17, 5);
```

#### Python
```python
# Python - tuples are natural
def divmod(a: int, b: int) -> tuple[int, int]:
    return a // b, a % b

quotient, remainder = divmod(17, 5)
```

---

## Classes and Objects

### Basic Class

#### TypeScript
```typescript
// TypeScript
class Dog {
  name: string;
  age: number;

  constructor(name: string, age: number) {
    this.name = name;
    this.age = age;
  }

  bark(): void {
    console.log(`${this.name} says woof!`);
  }
}

const myDog = new Dog("Buddy", 3);
myDog.bark();
```

#### Python
```python
# Python
class Dog:
    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age

    def bark(self) -> None:
        print(f"{self.name} says woof!")

my_dog = Dog("Buddy", 3)
my_dog.bark()
```

**Key Differences:**
1. `__init__` instead of `constructor`
2. `self` instead of `this` (and it's explicit in parameters!)
3. No `new` keyword - just call the class
4. Double underscores (`__init__`) are special methods ("dunder" methods)

### Class Properties and Methods

#### TypeScript
```typescript
class Person {
  // Private field
  private _age: number;

  constructor(name: string, age: number) {
    this.name = name;
    this._age = age;
  }

  // Getter
  get age(): number {
    return this._age;
  }

  // Setter
  set age(value: number) {
    if (value < 0) throw new Error("Invalid age");
    this._age = value;
  }
}
```

#### Python
```python
class Person:
    def __init__(self, name: str, age: int):
        self.name = name
        self._age = age  # Convention: _ means "private"

    @property
    def age(self) -> int:
        """Getter using @property decorator"""
        return self._age

    @age.setter
    def age(self, value: int) -> None:
        """Setter for age property"""
        if value < 0:
            raise ValueError("Invalid age")
        self._age = value

# Usage
person = Person("Alice", 30)
print(person.age)  # Uses getter
person.age = 31    # Uses setter
```

### Inheritance

Both languages support inheritance similarly:

#### TypeScript
```typescript
class Animal {
  speak(): void {
    console.log("Some sound");
  }
}

class Cat extends Animal {
  speak(): void {
    console.log("Meow");
  }
}
```

#### Python
```python
class Animal:
    def speak(self) -> None:
        print("Some sound")

class Cat(Animal):
    def speak(self) -> None:
        print("Meow")
```

**Real example from Price Stradamus:**
```python
# From src/price_stradamus/models/base.py
class BaseModel(ABC):
    """Abstract base class for all models"""

    @abstractmethod
    def fit(self, train_series: TimeSeries) -> None:
        pass

# From src/price_stradamus/models/neural/nbeats.py
@ModelRegistry.register("nbeats")
class NBEATSModel(BaseModel):
    def fit(self, train_series: TimeSeries) -> None:
        # Implementation here
        pass
```

---

## Type Hints (TypeScript-like)

Python's type hints are similar to TypeScript's type annotations.

### Basic Type Hints

```python
# Basic types
def add(a: int, b: int) -> int:
    return a + b

# String
def greet(name: str) -> str:
    return f"Hello, {name}"

# Boolean
def is_valid(value: float) -> bool:
    return value > 0
```

### Optional Types (Union with None)

#### TypeScript
```typescript
function findUser(id: string): User | null {
  // ...
}
```

#### Python (Python 3.10+ syntax)
```python
def find_user(id: str) -> User | None:
    # ...
    return None

# Older syntax (still common)
from typing import Optional

def find_user(id: str) -> Optional[User]:
    # ...
```

### Generic Types

#### TypeScript
```typescript
function first<T>(arr: T[]): T | undefined {
  return arr[0];
}
```

#### Python
```python
from typing import TypeVar

T = TypeVar('T')

def first(arr: list[T]) -> T | None:
    return arr[0] if arr else None
```

### Complex Types

```python
from typing import Dict, List, Tuple, Any

# List of strings (like string[] in TS)
names: list[str] = ["Alice", "Bob"]

# Dictionary (like Record<string, number> in TS)
scores: dict[str, int] = {"Alice": 95, "Bob": 87}

# Tuple with specific types
point: tuple[float, float] = (10.5, 20.3)

# Any type (avoid when possible)
data: Any = "could be anything"
```

**Real example from Price Stradamus:**
```python
# From src/price_stradamus/data/features.py
def generate_all_features(
    self,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Generate all technical indicators.

    Args:
        df: DataFrame with OHLCV columns

    Returns:
        DataFrame with added feature columns
    """
    ...
```

### Pydantic for Data Validation

Price Stradamus uses **Pydantic** for runtime type validation (like Zod in TypeScript):

#### TypeScript + Zod
```typescript
import { z } from 'zod';

const UserSchema = z.object({
  name: z.string(),
  age: z.number().positive(),
  email: z.string().email(),
});

type User = z.infer<typeof UserSchema>;
```

#### Python + Pydantic
```python
from pydantic import BaseModel, Field, EmailStr

class User(BaseModel):
    name: str
    age: int = Field(gt=0)
    email: EmailStr

# Usage
user = User(name="Alice", age=25, email="alice@example.com")
# Validates at runtime!
```

**Real example from Price Stradamus:**
```python
# From src/price_stradamus/config/settings.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    database_url: str = Field(..., description="PostgreSQL connection URL")
    default_symbol: str = "BTCUSDT"
    default_timeframe: str = "1m"

    class Config:
        env_file = ".env"
```

---

## Collections and Iteration

### Lists (Arrays)

#### TypeScript
```typescript
const numbers: number[] = [1, 2, 3, 4, 5];
numbers.push(6);
const doubled = numbers.map(n => n * 2);
const evens = numbers.filter(n => n % 2 === 0);
```

#### Python
```python
numbers: list[int] = [1, 2, 3, 4, 5]
numbers.append(6)
doubled = [n * 2 for n in numbers]  # List comprehension
evens = [n for n in numbers if n % 2 == 0]
```

### Dictionaries (Objects)

#### TypeScript
```typescript
const person: Record<string, any> = {
  name: "Alice",
  age: 30,
};

console.log(person.name);
console.log(person["age"]);
```

#### Python
```python
person: dict[str, any] = {
    "name": "Alice",
    "age": 30,
}

print(person["name"])  # Dictionary access
# No person.name syntax (use dataclasses or Pydantic for that)
```

### Iteration

#### TypeScript
```typescript
// for...of loop
for (const num of numbers) {
  console.log(num);
}

// forEach
numbers.forEach((num, index) => {
  console.log(index, num);
});
```

#### Python
```python
# for loop (no 'of', no 'const')
for num in numbers:
    print(num)

# enumerate for index
for index, num in enumerate(numbers):
    print(index, num)
```

---

## Comprehensions

Python's comprehensions are powerful one-liners (no direct TS equivalent).

### List Comprehension

```python
# Instead of this:
squares = []
for x in range(10):
    squares.append(x ** 2)

# Write this:
squares = [x ** 2 for x in range(10)]

# With condition
evens = [x for x in range(10) if x % 2 == 0]
```

**TypeScript comparison:**
```typescript
// TypeScript equivalent
const squares = Array.from({length: 10}, (_, x) => x ** 2);
const evens = Array.from({length: 10}, (_, x) => x).filter(x => x % 2 === 0);
```

### Dictionary Comprehension

```python
# Create dict from two lists
keys = ["a", "b", "c"]
values = [1, 2, 3]
mapping = {k: v for k, v in zip(keys, values)}
# Result: {"a": 1, "b": 2, "c": 3}

# Transform dictionary
prices = {"BTC": 50000, "ETH": 3000}
prices_k = {k: v / 1000 for k, v in prices.items()}
# Result: {"BTC": 50, "ETH": 3}
```

**Real example from Price Stradamus:**
```python
# From src/price_stradamus/evaluation/metrics.py
def calculate_all(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float]:
    return {
        "mae": np.mean(np.abs(actual - predicted)),
        "rmse": np.sqrt(np.mean((actual - predicted) ** 2)),
        "mape": np.mean(np.abs((actual - predicted) / actual)) * 100,
    }
```

---

## Decorators

Decorators are like TypeScript decorators, but used more commonly in Python.

### Basic Decorator

```python
def log_call(func):
    """Decorator that logs function calls"""
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__}")
        result = func(*args, **kwargs)
        print(f"{func.__name__} returned {result}")
        return result
    return wrapper

@log_call
def add(a: int, b: int) -> int:
    return a + b

result = add(2, 3)
# Prints: Calling add
#         add returned 5
```

**TypeScript comparison:**
```typescript
function logCall(target: any, propertyName: string, descriptor: PropertyDescriptor) {
  const originalMethod = descriptor.value;
  descriptor.value = function(...args: any[]) {
    console.log(`Calling ${propertyName}`);
    const result = originalMethod.apply(this, args);
    console.log(`${propertyName} returned ${result}`);
    return result;
  };
}

class Calculator {
  @logCall
  add(a: number, b: number): number {
    return a + b;
  }
}
```

### Common Decorators in Price Stradamus

#### 1. @property (getter/setter)
```python
class Model:
    def __init__(self):
        self._is_trained = False

    @property
    def is_trained(self) -> bool:
        return self._is_trained
```

#### 2. @abstractmethod (interface enforcement)
```python
from abc import ABC, abstractmethod

class BaseModel(ABC):
    @abstractmethod
    def fit(self, data):
        pass  # Must be implemented by subclasses
```

#### 3. @staticmethod and @classmethod
```python
class MathHelper:
    @staticmethod
    def add(a: int, b: int) -> int:
        return a + b  # No self needed

    @classmethod
    def from_config(cls, config: dict):
        return cls(**config)  # cls is the class itself
```

#### 4. Custom decorator in Price Stradamus
```python
# Model registry decorator
@ModelRegistry.register("nbeats")
class NBEATSModel(BaseModel):
    ...
```

---

## Modules and Imports

### Import Syntax

#### TypeScript
```typescript
// TypeScript
import { something } from './module';
import * as all from './module';
import defaultExport from './module';
```

#### Python
```python
# Python
from module import something
import module as all
from module import default_export

# Multiple imports
from module import func1, func2, Class1
```

### Real Examples from Price Stradamus

```python
# Standard library
from __future__ import annotations
import asyncio
from pathlib import Path
from datetime import datetime

# Third-party packages
import pandas as pd
import torch
from darts import TimeSeries

# Local imports (relative)
from price_stradamus.config import settings
from price_stradamus.utils import logger
from price_stradamus.models.base import BaseModel
```

### Package Structure

```
src/price_stradamus/
├── __init__.py          # Makes directory a package
├── config/
│   ├── __init__.py
│   └── settings.py
└── models/
    ├── __init__.py
    ├── base.py
    └── neural/
        ├── __init__.py
        └── nbeats.py
```

To import:
```python
from price_stradamus.models.neural.nbeats import NBEATSModel
```

---

## Error Handling

### Try/Catch vs Try/Except

#### TypeScript
```typescript
try {
  const result = riskyOperation();
} catch (error) {
  console.error("Error:", error);
} finally {
  cleanup();
}
```

#### Python
```python
try:
    result = risky_operation()
except Exception as error:
    print(f"Error: {error}")
finally:
    cleanup()
```

### Specific Exception Types

```python
try:
    value = int(input("Enter a number: "))
    result = 10 / value
except ValueError:
    print("Invalid number")
except ZeroDivisionError:
    print("Cannot divide by zero")
except Exception as e:
    print(f"Unexpected error: {e}")
finally:
    print("Cleanup")
```

### Custom Exceptions

```python
class DataFetchError(Exception):
    """Raised when data fetching fails"""
    pass

class ModelTrainingError(Exception):
    """Raised during model training"""
    pass

# Usage
try:
    data = fetch_binance_data()
    if data is None:
        raise DataFetchError("No data received")
except DataFetchError as e:
    logger.error(f"Failed to fetch data: {e}")
```

**Real example from Price Stradamus:**
```python
# From src/price_stradamus/data/fetcher.py
async def fetch_historical_range(
    self,
    symbol: str,
    interval: str,
    start_date: datetime,
    end_date: datetime,
) -> pd.DataFrame:
    try:
        # Fetch data
        data = await self._fetch_batch(symbol, interval, start_ms, end_ms)
        return data
    except aiohttp.ClientError as e:
        raise DataFetchError(f"Failed to fetch {symbol}") from e
```

---

## Quick Reference

### Syntax Cheat Sheet

```python
# Variables
name: str = "Alice"
age: int = 25
price: float = 99.99
is_active: bool = True
nothing: None = None

# Functions
def greet(name: str) -> str:
    return f"Hello, {name}"

# Classes
class Dog:
    def __init__(self, name: str):
        self.name = name

    def bark(self) -> None:
        print(f"{self.name} barks!")

# Lists
numbers: list[int] = [1, 2, 3]
numbers.append(4)

# Dictionaries
person: dict[str, any] = {"name": "Alice", "age": 30}

# Loops
for num in numbers:
    print(num)

# Comprehensions
squares = [x ** 2 for x in range(10)]

# Conditional
if age >= 18:
    print("Adult")
elif age >= 13:
    print("Teen")
else:
    print("Child")

# Error handling
try:
    result = risky_operation()
except Exception as e:
    print(f"Error: {e}")
```

### TypeScript → Python Quick Map

| TypeScript | Python | Notes |
|-----------|--------|-------|
| `let x = 5` | `x = 5` | No let/const |
| `const x = 5` | `x = 5` | Convention: UPPERCASE for constants |
| `null`, `undefined` | `None` | Single null value |
| `true`, `false` | `True`, `False` | Capitalized |
| `console.log()` | `print()` | Built-in print |
| `function` | `def` | Function definition |
| `() => {}` | `lambda` | For simple functions only |
| `class` | `class` | Similar |
| `constructor` | `__init__` | Constructor method |
| `this` | `self` | Explicit self parameter |
| `extends` | `(Parent)` | Inheritance syntax |
| `import { x }` | `from module import x` | Import syntax |
| `try/catch` | `try/except` | Error handling |

---

## Practice Exercises

### Exercise 1: Hello Python (10 minutes)

Convert this TypeScript to Python:

```typescript
function calculateDiscount(price: number, percentage: number): number {
  return price * (1 - percentage / 100);
}

const originalPrice = 100;
const discountedPrice = calculateDiscount(originalPrice, 20);
console.log(`Discounted price: $${discountedPrice}`);
```

<details>
<summary>Solution</summary>

```python
def calculate_discount(price: float, percentage: float) -> float:
    return price * (1 - percentage / 100)

original_price = 100
discounted_price = calculate_discount(original_price, 20)
print(f"Discounted price: ${discounted_price}")
```
</details>

---

### Exercise 2: Classes (15 minutes)

Convert this TypeScript class to Python:

```typescript
class BankAccount {
  private balance: number;

  constructor(initialBalance: number) {
    this.balance = initialBalance;
  }

  deposit(amount: number): void {
    this.balance += amount;
  }

  withdraw(amount: number): boolean {
    if (amount > this.balance) {
      return false;
    }
    this.balance -= amount;
    return true;
  }

  getBalance(): number {
    return this.balance;
  }
}
```

<details>
<summary>Solution</summary>

```python
class BankAccount:
    def __init__(self, initial_balance: float):
        self._balance = initial_balance

    def deposit(self, amount: float) -> None:
        self._balance += amount

    def withdraw(self, amount: float) -> bool:
        if amount > self._balance:
            return False
        self._balance -= amount
        return True

    @property
    def balance(self) -> float:
        return self._balance
```
</details>

---

### Exercise 3: List Comprehensions (10 minutes)

Rewrite these using Python list comprehensions:

```typescript
// 1. Square all numbers
const numbers = [1, 2, 3, 4, 5];
const squares = numbers.map(n => n ** 2);

// 2. Filter even numbers
const evens = numbers.filter(n => n % 2 === 0);

// 3. Square only even numbers
const evenSquares = numbers
  .filter(n => n % 2 === 0)
  .map(n => n ** 2);
```

<details>
<summary>Solution</summary>

```python
# 1. Square all numbers
numbers = [1, 2, 3, 4, 5]
squares = [n ** 2 for n in numbers]

# 2. Filter even numbers
evens = [n for n in numbers if n % 2 == 0]

# 3. Square only even numbers
even_squares = [n ** 2 for n in numbers if n % 2 == 0]
```
</details>

---

### Exercise 4: Type Hints (15 minutes)

Add proper type hints to this Python code:

```python
def fetch_user_data(user_id):
    # Simulates fetching user data
    if user_id == 1:
        return {"name": "Alice", "age": 30, "email": "alice@example.com"}
    return None

def process_users(user_ids):
    results = []
    for user_id in user_ids:
        user = fetch_user_data(user_id)
        if user:
            results.append(user)
    return results
```

<details>
<summary>Solution</summary>

```python
from typing import TypedDict

class User(TypedDict):
    name: str
    age: int
    email: str

def fetch_user_data(user_id: int) -> User | None:
    """Fetch user data by ID."""
    if user_id == 1:
        return {"name": "Alice", "age": 30, "email": "alice@example.com"}
    return None

def process_users(user_ids: list[int]) -> list[User]:
    """Process multiple user IDs and return user data."""
    results: list[User] = []
    for user_id in user_ids:
        user = fetch_user_data(user_id)
        if user:
            results.append(user)
    return results
```
</details>

---

## Next Steps

Congratulations! You now understand Python basics from a TypeScript perspective.

**Next Module:** [02: Async & Data Structures →](02-async-and-data-structures.md)

In the next module, you'll learn async/await patterns (similar to Node.js) and work with pandas/numpy (the data manipulation libraries).

---

## Summary Checklist

- [ ] I can read Python syntax without confusion
- [ ] I understand indentation-based blocks
- [ ] I know how to use type hints
- [ ] I can write classes with `__init__` and `self`
- [ ] I understand list/dict comprehensions
- [ ] I know the difference between `None` and `null`/`undefined`
- [ ] I can use decorators (@property, @staticmethod)
- [ ] I understand Python imports

**Ready?** Continue to [Module 02: Async & Data Structures](02-async-and-data-structures.md)!
