![Alt text](https://github.com/xdityagr/Fusion/blob/main/media/banner_fusion.png?raw=true "Banner Image")

# **Fusion** Programming Language

**Fusion (*.fs) [Version 0.1.0-alpha]**

Fusion is a programming language that aims to blends Python's simplicity, C++'s speed, Rust's safety, and Go's concurrency to deliver high-performance and clear code. 
Started to be developed to explore interpreter design and how compilers works and now this personal learning project turns into a professional project that impacts the real world aiming Fusion ignites a new era of programming and answer the question, *What programming language should i learn first?*

## Overview

Fusion iscurrently in the development phase with a focus on building a robust interpreter and core language features and will be converted into a compiled language using LLVM, as soon as the author learns to deal with compilers.

Currently, Fusion aims to combines the best features of modern languages:

- **Python's Simplicity**: Easy-to-read syntax for rapid development with a new fresh syntax.
- **C++'s Speed**: Optimized for performance.
- **Rust's Safety**: Strong type system and memory safety.
- **Go's Concurrency**: Built-in support for concurrent programming.


## Features

- **Dynamic Typing with Optional Type Annotations**: Declare variables with or without types (e.g., `let x = 5` or `let x : Int = 5`).
- **Control Structures**: Supports `if`, `elif`, `else`, `for`, `while`, and more.
- **Functions**: Define named and anonymous functions (e.g., `fun add(a, b) { a + b }` or `~a,b { a + b }`).
- **Built-in Functions**: Includes `print`, `input`, `run`,  `clear`, `exit`, and more.
- **Lists and Strings**: Supports list and string operations, including slicing (e.g., `[1, 2, 3][0..2]`).
- **Error Handling**: Detailed error messages with tracebacks for debugging.

## Installation

To get started with Fusion, download the executable from the releases page:

1. **Download Fusion**:
   - Visit the [Latest Releases Page](https://github.com/xdityagr/Fusion/releases/tag/v0.1.0-alpha).
   - Download the latest `Fusion.exe` for your operating system.

2. **Run Fusion**:
   - Place `Fusion.exe` in your desired directory.
   - Open a terminal in that directory and run:
     ```bash
     Fusion
     ```
     This will launch the Fusion Shell (Version 0.1.0 Dev).

3. **Run a Fusion File**:
   - Save your code in a `.fs` file (e.g., `main.fs`).
   - Run:
     ```bash
     Fusion main.fs
     ```

## Usage

### Interactive Shell
Launch the Fusion Shell to experiment with code interactively. 
Here's a simple example :
```plaintext
Fusion 0.1.0-alpha (dev, Mar 24 2025) [Py 3.13.2, 64 bit (x86_64)] on win32
>>> let start = input("Enter countdown start (e.g., 5): ")
Enter countdown start (e.g., 5): 3
>>> let count = 3  # Assuming input is "3"
>>> for i in count..0..-1 { print(i); if i == 0 { print("Liftoff! 🚀") } }

3
2
1
0
Liftoff! 🚀
```

### Writing a Fusion Program
Create a file `rpg_game.fs` to simulate a simple text-based RPG battle, showcasing variable definitions, loops, conditionals, functions, and list operations:

```swift
# RPG Battle Simulator
let player_hp = 100
let enemy_hp = 80
let player_attacks = ["Fireball", "Ice Slash", "Thunder Strike"]

# Function to calculate damage with a random factor
fun calculate_damage(attack) {
    let base_dmg = if attack == "Fireball" { 20 } elif attack == "Ice Slash" { 15 } else { 25 }
    let random_factor = 5  # Simplified for demo
    base_dmg + random_factor
}

print("=== RPG Battle Start ===")
print("Player HP: " + player_hp + " | Enemy HP: " + enemy_hp)

# Battle loop: alternate turns until someone wins
let turn = 0
while player_hp > 0 and enemy_hp > 0 {
    let attack_idx = turn % 3  # Cycle through attacks
    let attack = player_attacks[attack_idx]
    let damage = calculate_damage(attack)

    # Player's turn
    print("Player uses " + attack + " dealing " + damage + " damage!")
    enemy_hp = enemy_hp - damage

    # Enemy's turn (simple counterattack)
    if enemy_hp > 0 {
        let enemy_dmg = 10
        print("Enemy counterattacks for " + enemy_dmg + " damage!")
        player_hp = player_hp - enemy_dmg
    }

    print("Player HP: " + player_hp + " | Enemy HP: " + enemy_hp)
    turn = turn + 1
}

# Determine the winner
let result = if player_hp > 0 { "Player wins the battle!" } else { "Enemy wins the battle!" }
print(result)
print("=== Battle End ===")
```

Run it:
```bash
fusion.exe rpg_game.fs
```

**Output**:
```plaintext
=== RPG Battle Start ===
Player HP: 100 | Enemy HP: 80
Player uses Fireball dealing 25 damage!
Enemy counterattacks for 10 damage!
Player HP: 90 | Enemy HP: 55
Player uses Ice Slash dealing 20 damage!
Enemy counterattacks for 10 damage!
Player HP: 80 | Enemy HP: 35
Player uses Thunder Strike dealing 30 damage!
Enemy counterattacks for 10 damage!
Player HP: 70 | Enemy HP: 5
Player uses Fireball dealing 25 damage!
Player HP: 70 | Enemy HP: -20
Player wins the battle!
=== Battle End ===
```

## Syntax Highlights

- **Variable Declaration**:
  ```swift
  let score = 0
  let name:String = "Alex"
  ```
- **Conditionals**:
  ```swift
  let status = if score > 50 { "Pass" } else { "Fail" }
  ```
- **Loops**:
  ```swift
  for i in 1..5 { print("Round " + i) }  # Prints Round 1 to Round 5
  while score < 100 { score = score + 10; print(score) }
  ```
- **Functions**:
  ```swift
  fun double(x) { x * 2 }
  print(double(5))  # Prints 10
  ```
- **Lists and Slicing**:
  ```swift
  let items = ["sword", "shield", "potion"]
  print(items[0..2])  # Prints ["sword", "shield"]
  ```

## Development Status & Future Goals 

Fusion is in active development (Version 0.1.0-alpha). Current features include a lexer, parser, and interpreter with support for basic constructs. Future updates will focus on:
- Covertion to compiled language using LLVM
- Concurrency inspired by Go.
- Performance optimizations for speed.
- Standard library expansion.

## Issues and Feedback

Found a bug or have a suggestion? Open an issue on the [GitHub Issues page](https://github.com/xdityagr/fusion/issues).

## Author

Developed by **Aditya Gaur**, With love in ***India <3***
- GitHub: [xdityagr](https://github.com/xdityagr)  
- Contact: [adityagaur.home@gmail.com]()

## License

Fusion is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
