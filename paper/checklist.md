
### introduction

- [x] describe boundary modulation (BND) in term of possible actual neuromodulators, e.g. serotonin

### discussion

---
- [x] I would also emphasize more strongly in the introduction and discussion the use of idiothetic information (path integration) as a strength and point of differentiation from other models relying on visual cues etc.
- [x] Frame this as a deliberate choice to isolate the mechanisms of online map formation driven by internal state.

The model uses separate neuromodulators for rewards (DA) and boundaries (BND).
- [x] You could predict that pharmacologically blocking dopamine signaling in the hippocampus during a navigation task would selectively impair the agent's ability to update reward locations, while having a less pronounced effect on its ability to learn and avoid new obstacles. This is a more specific and falsifiable hypothesis than a general alignment.

---
In the discussion, add a paragraph addressing the choice of a Dijkstra-like path-finding algorithm operating on the place cell graph.
- [x] Justify why this abstraction was made (e.g., to focus on the map formation process rather than the planning dynamics).
- [x] More importantly, discuss how a more biologically plausible mechanism, such as hippocampal replay or spreading activation within the place cell network, could approximate the function of this algorithm.

This shows awareness of the model's limitations and connects it back to ongoing biological research.

---
The model shows that decreasing place field size (by increasing gain) at boundaries is crucial for performance.
- [x] You could predict that optogenetically manipulating the excitability of CA1 place cells specifically at environmental boundaries during exploration would directly impact the precision of the resulting spatial map and subsequent navigation efficiency.

---
- [x] Add a paragraph to the discussion that explicitly compares and contrasts your model with the SR framework. For example, you could argue that while SRs elegantly handle goal-directed navigation through a predictive map, your model provides a more explicit mechanism for how environmental boundaries (via BND modulation) and salient events dynamically and locally shape the structure of the map itself (via gain and density modulation), which is a different emphasis.

---
- [ ] limitations | hm, what more

---
- [x] compilation error
