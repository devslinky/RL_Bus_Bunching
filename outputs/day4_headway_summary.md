1. Why the placeholder (headway ~ (300,60)) is dangerously wrong and what it breaks downstream?

The true/real headway is ~ (536,296). Therefore the placeholder is wildly innacurate; this has downstream affects on stop arrival rates since we calculate arrival rate by # boardings / headway. 
