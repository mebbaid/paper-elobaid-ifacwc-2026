## Stabilization of Rigid Formations on Regular Curves

This repository contains the reference implementation for the paper **"On the Stabilization of Rigid Formations on Regular Curves"**. 

It provides a unified control framework that enables multi-agent systems to:
1.  Find inscribed rigid polygon formations centered on a target on general planar curves (including those with cusps and self-intersections) via a randomized multi-start optimization.
2.  Control agents to sweep the path and smoothly transition to the desired formation vertices.
3.  Ensure inter-agent collision avoidance and curve "practical invariance" throughout the mission.

The mission scenario demonstrates a transition from coordinated continuous coverage (sweeping) to targeted inspection (formation stabilization).


## Quick start

This project uses the `pixi` tool for dependency management and convenient run scripts (optional).

Installation (pixi):

```bash
# Install pixi (optional)
curl -fsSL https://pixi.sh/install.sh | bash
```

Run examples:

```bash
# Reporduce paper simulation
pixi run simulatation-example

# Reporduce paper Example 3
pixi run finder-example

```


![Multi-agent formation control simulation](assets/multi_agent_formation.gif)

## License

BSD3 License 

---


## Maintainers

<table align="left">
    <tr>
        <td><a href="https://github.com/mebbaid"><img src="https://github.com/mebbaid.png" width="40"></a></td>
        <td><a href="https://github.com/mebbaid">👨‍💻 @Mohamed Elobaid</a></td>
    </tr>
</table>

