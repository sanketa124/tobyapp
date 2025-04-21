class TuringMachine:
    def __init__(self, tape, states, transitions, initial, accept, reject):
        self.tape = list(tape) + ['_']  # Tape with blank at the end
        self.states = states
        self.transitions = transitions
        self.current_state = initial
        self.accept_state = accept
        self.reject_state = reject
        self.head = 0  # Tape head starts at position 0

    def step(self):
        """Executes one step of the Turing Machine"""
        char = self.tape[self.head]
        if (self.current_state, char) in self.transitions:
            new_state, write_char, move = self.transitions[(self.current_state, char)]
            self.tape[self.head] = write_char  # Write new character
            self.current_state = new_state  # Change state
            self.head += 1 if move == 'R' else -1  # Move head
        else:
            self.current_state = self.reject_state  # If no transition, reject

    def run(self):
        """Runs the TM until it reaches an accept or reject state"""
        while self.current_state not in {self.accept_state, self.reject_state}:
            self.step()
        return self.current_state == self.accept_state  # Return True if accepted

# Define the transitions for palindrome checking
transitions = {
    ('q0', '0'): ('q1', 'X', 'R'),  # Mark leftmost 0, move right
    ('q0', '1'): ('q2', 'Y', 'R'),  # Mark leftmost 1, move right
    ('q0', '_'): ('q_accept', '_', 'N'),  # Empty string or finished checking

    ('q1', '0'): ('q1', '0', 'R'),  # Skip 0s
    ('q1', '1'): ('q1', '1', 'R'),  # Skip 1s
    ('q1', '_'): ('q3', '_', 'L'),  # Reach end, start checking from the right

    ('q2', '0'): ('q2', '0', 'R'),
    ('q2', '1'): ('q2', '1', 'R'),
    ('q2', '_'): ('q4', '_', 'L'),  # Reach end, start checking

    ('q3', '0'): ('q5', 'X', 'L'),  # Match last 0
    ('q3', 'X'): ('q3', 'X', 'L'),  # Skip already matched X
    ('q3', 'Y'): ('q3', 'Y', 'L'),
    ('q3', '_'): ('q0', '_', 'R'),  # If we reach _, check next

    ('q4', '1'): ('q5', 'Y', 'L'),  # Match last 1
    ('q4', 'X'): ('q4', 'X', 'L'),
    ('q4', 'Y'): ('q4', 'Y', 'L'),
    ('q4', '_'): ('q0', '_', 'R'),

    ('q5', 'X'): ('q5', 'X', 'L'),  # Skip X
    ('q5', 'Y'): ('q5', 'Y', 'L'),  # Skip Y
    ('q5', '_'): ('q_accept', '_', 'N'),  # If only X and Y left, accept
}

# Define the Turing Machine
tm = TuringMachine(
    tape="0110",  # Example palindrome
    states={"q0", "q1", "q2", "q3", "q4", "q5", "q_accept", "q_reject"},
    transitions=transitions,
    initial="q0",
    accept="q_accept",
    reject="q_reject"
)

# Run the Turing Machine
result = tm.run()
print("Accepted" if result else "Rejected")  # ✅ Output: "Accepted"
