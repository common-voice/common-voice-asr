# Training Red Flag Notes
- Loss remaining stagnant at ~2.8-3.0 after dipping dramatically from the beginning of the first epoch
- Words repeating without correlation to actual reference values
    Run exalted-sweep-6:
        Ref ['IT', 'HAS', 'ALSO', 'BEEN', 'PROPOSED', 'AS', 'A', 'POSSIBLE', 'EXPLANATION', 'FOR', 'CLAIMS', 'ASSOCIATED', 'WITH', 'STIGMATA'] Hypo: ['HER'] Ref ['CABLE', 'ENDS', 'UP', 'SHOOTING', 'HAMMER', 'PARALYZING', 'HIM'] Hypo: ['HER']
    Run pleasant-sweep-5:
        Ref ['WEATHER', 'FORECASTING', 'IS', 'ANOTHER', 'CRITICAL', 'ASPECT', 'OF', 'SAILING', 'YACHT', 'MANAGEMENT'] Hypo: ['THEIR'] Ref ['THE', 'BOAT', 'WAS', 'SINKING', 'AND', 'THE', 'CREW', 'WAS', 'ABANDONING', 'SHIP'] Hypo: ['THEIR'] (seriously so much there)
    Run playful-sweep-4:
        Completely empty hypotheses until the final epoch, there it repeated 'TENS'
    Run peach-sweep-7: 
        Whole lot of 'THEY'
    Run azure-sweep-8: did not come up with any hypothesis throughout
    Run expert-sweep-12: empty/'THE', 'AND', 'EACH' throughout
    Run kind-sweep-14:
        variation in the first epoch, slowly descented to just be 'EACH' & 'SEE' in 3/4, 'EYES' & one 'SEE' in 5
    Run atomic-sweep-29:
        best thus far, whole lot of 'OF' - seems to be tracking most likely values rather than actually using the input 
    Run sage-sweep-41:
        epoch 1 includes actual guesses & empty, but then goes entirely empty hypotheses
    Run different-sweep-43:
        Val WER: 0.9472 after first epoch, interesting because lot of empty vals in the print during train. WER got worse with each epoch, going back to .99 for 2 & 3, then .97/.98 for the rest
    Run royal-sweep-47
        Lot of 'THE' & 'THEY'. Epoch 1: Val WER: 0.9592, Epoch 2: 0.9517 Epoch 3: 0.9517, Epoch 4: 0.9517, Epoch 5: Val WER: 0.9802
    Run azure-sweep-53: 
        High loss throughout, going from 10-6-4
    Run soft-sweep-54:
        totally empty hypotheses by the end
    Run cerulean-sweep-55:
        high loss, empty hypotheses towards the end
    Run magic-sweep-85:
        totally empty