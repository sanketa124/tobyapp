from automata.tm.dtm import DTM

# Define a Linear Bounded Automaton (LBA) which is a restricted TM
lba = DTM(
    states={"q0", "q1", "q2", "q3", "q_accept"},
    input_symbols={"a", "b", "c"},
    tape_symbols={"a", "b", "c", "_"},
    transitions={
        ("q0", "a"): ("q0", "a", "R"),
        ("q0", "b"): ("q1", "b", "R"),
        ("q1", "b"): ("q1", "b", "R"),
        ("q1", "c"): ("q2", "c", "R"),
        ("q2", "c"): ("q2", "c", "R"),
        ("q2", "_"): ("q_accept", "_", "N"),
    },
    initial_state="q0",
    blank_symbol="_",
    final_states={"q_accept"},
)

print(lba.accepts_input("aabbcc"))  # ✅ True
print(lba.accepts_input("aabbbc"))  # ❌ False
