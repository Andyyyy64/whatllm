# VHS demo recording for whichllm
# Install: https://github.com/charmbracelet/vhs
# Run: vhs demo.tape
# Output: assets/demo.gif

Output assets/demo.gif

Set FontSize 15
Set Width 1200
Set Height 700
Set Theme "Catppuccin Mocha"
Set Padding 20
Set TypingSpeed 50ms

# Warm up cache invisibly
Hide
Type "whichllm --gpu 'RTX 3060' --top 1 > /dev/null 2>&1"
Enter
Sleep 10s
Type "clear"
Enter
Sleep 500ms
Show

# Scene 1: Basic usage - "what can I run?"
Type "whichllm --gpu 'RTX 3060' --top 5"
Sleep 500ms
Enter
Sleep 8s

# Pause to read results
Sleep 5s

# Scene 2: Plan command - "what do I need for this model?"
Type "whichllm plan 'qwen 72b'"
Sleep 500ms
Enter
Sleep 8s

# Pause to read results
Sleep 5s

# Scene 3: Profile-based - coding focus
Type "whichllm --gpu 'RTX 3060' --profile coding --top 3"
Sleep 500ms
Enter
Sleep 8s

# Final pause
Sleep 4s
