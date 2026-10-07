# A Philosophy of Software Design: a compact summary

John Ousterhout, *A Philosophy of Software Design* (2nd ed., 2021). This is a summary of
its ideas and of what they imply in general. What they meant for this repository is in
`decisions.md`.

## 1. The central claim

The main problem in software design is **complexity**: anything about a system's
structure that makes it hard to understand or change.

- **Symptoms:**
  - *Change amplification:* a simple change touches many places.
  - *Cognitive load:* how much a developer must know to make the change.
  - *Unknown unknowns:* it isn't clear what must change, or what knowledge is needed.
    This is the worst of the three.
- **Causes:**
  - *Dependencies:* code that can't be understood or changed in isolation.
  - *Obscurity:* important information that isn't obvious.
- **Complexity is incremental.** No single shortcut makes a system complex; hundreds
  of small ones do. So zero tolerance is the only stable policy.

## 2. Strategic vs. tactical programming

- **Tactical programming** aims to get the current feature working as fast as
  possible. Every task adds a little complexity. The "tactical tornado" is fast in
  the short term and leaves a mess.
- **Strategic programming** treats working code as necessary but not enough. The
  goal is a good design that also works. Invest about 10–20% of time in design
  continually, not in a later cleanup phase that never comes.

## 3. Modules should be deep

- A module (class, function, file, package, CLI tool) has an **interface**, the
  cost it imposes on users, and an **implementation**, the functionality it gives
  them.
- **Deep module:** a simple interface over substantial functionality. The classic
  example is Unix file I/O: five calls hide disks, caches, permissions and
  scheduling.
- **Shallow module:** an interface nearly as complex as what it does. Pass-through
  wrappers, tiny classes and one-line helpers with long signatures are examples.
  They add interface cost and give little in return.
- **"Classitis":** the belief that more, smaller classes or functions are always
  better. Splitting has a cost: every new interface is something to learn.
- The **interface includes informal parts**: side effects, ordering constraints,
  required config keys, file-layout conventions. These count toward complexity
  even though no type signature shows them.

## 4. Information hiding (and leakage)

- Each module should hide **design decisions** (data formats, algorithms, file
  layouts, naming conventions) so that a change in one decision changes only one
  module.
- **Information leakage** is the opposite: one decision is reflected in several
  modules. For example, two scripts that both know the output directory layout, or
  a metric name spelled as a string literal in a trainer, an analysis script and a
  config.
- **Temporal decomposition** is a common cause of leakage: structuring code by the
  order things happen (load, then train, then evaluate, then write) instead of by
  what knowledge each part encapsulates. Each phase then needs to know the same
  formats.
- **Overexposure:** make the common case simple. Rarely-used options shouldn't be
  forced on every caller; good defaults are a form of information hiding.

## 5. General-purpose modules are deeper

- Aim for **"somewhat general-purpose"**: the interface is general enough for
  several uses, while the implementation serves today's needs.
- Questions to ask:
  - What is the simplest interface that covers all current uses?
  - In how many situations will this method be used?
  - Is this API easy to use for my current needs?
- **Push specialization upward or downward.** Special cases belong in the top layer
  (scripts, configs) or the bottom layer (drivers, low-level primitives), not
  interleaved through the middle.

## 6. Different layer, different abstraction

- Adjacent layers that offer the **same abstraction** are a red flag. Examples are
  pass-through methods, pass-through variables threaded through many signatures,
  and decorators that add little.
- Instead of passing a variable through many layers, use a shared **context
  object**, or reconsider the split.

## 7. Pull complexity downwards

- It's better for a module's *implementation* to be complex than for its
  *interface* to be. Users outnumber implementers.
- **Configuration parameters are often complexity pushed upward.** Each knob says
  "I couldn't decide, you decide". Before adding one, ask whether the module can
  compute a good value itself. Expose a parameter only when callers truly have
  better information.

## 8. Better together or better apart?

- **Combine** pieces when they share information, are always used together,
  overlap conceptually, or when combining simplifies the interface or removes
  duplication.
- **Separate** general-purpose from special-purpose code.
- Splitting a long method is only worthwhile if the pieces are independently
  understandable. **Conjoined methods**, where you must read both to understand
  either, are a red flag. Length alone is not.

## 9. Define errors out of existence

- Exceptions and special cases inflate interfaces. Where possible, **define
  semantics so the error can't occur**. For example, deleting something that's
  already gone is a no-op, and taking a substring past the end gives a shorter
  string.
- Alternatives:
  - **Mask** errors at a low level.
  - **Aggregate** them in one handler higher up.
  - **Just crash** on errors that are rare and not worth handling.
- Design special cases out by making the normal path handle them.

## 10. Design it twice

- For any important module, sketch at least two **radically different** designs and
  compare them. The first idea is rarely the best.

## 11. Comments and documentation

- Comments capture what the code **cannot** say: intent, rationale, units,
  invariants and abstractions. Code says *what* and *how*; comments say *why*, and
  what a user must know.
- **Interface comments** describe the abstraction without the implementation.
  **Implementation comments** explain non-obvious *what* and *why*.
- **Write comments first.** They are a design tool: if an interface comment is
  hard to write, the interface is probably wrong.
- Comments that repeat the code are noise. **Cross-module decisions** need one
  canonical place, which the other places reference.

## 12. Names, consistency, obviousness

- Names should be **precise** and **consistent**: the same name always means the
  same thing, and different things get different names. If a name is hard to
  choose, that's a design smell.
- **Consistency** (naming, conventions, invariants) lets knowledge transfer: learn
  something once, apply it everywhere.
- **Code should be obvious.** A reader's first guess should be correct. Generic
  containers (tuples, `dict`s with implicit keys), event-driven control flow and
  hidden conventions all work against this.

## 13. Modifying existing code

- Stay strategic during maintenance. After a change, the system should look as if
  it had been **designed with that change in mind**, not have the change bolted on.
- Keep comments next to the code they describe, and update them with it.

## 14. Red flags (checklist)

| Red flag | Meaning |
|---|---|
| Shallow module | Interface not much simpler than implementation |
| Information leakage | One design decision reflected in several modules |
| Temporal decomposition | Structure follows execution order, not knowledge |
| Overexposure | Common use forces awareness of rare features |
| Pass-through method/variable | Layer adds nothing; argument threaded through |
| Repetition | Same nontrivial code in several places |
| Special–general mixture | Special-purpose code tangled into general modules |
| Conjoined methods | Can't understand one without the other |
| Comment repeats code | Comment adds no information |
| Implementation in interface docs | Interface comment leaks internals |
| Vague name | Name too imprecise to convey meaning |
| Hard to pick name | Underlying design is muddled |
| Hard to describe | Interface comment needs to be long, so the abstraction is wrong |
| Nonobvious code | Behaviour or meaning not apparent on reading |

## 15. General implications (beyond any one codebase)

1. **Count interfaces, not lines.** A codebase's complexity is driven by how many
   things a developer must know to make a change: modules, flags, config keys,
   conventions. Deleting a tool or a config knob often reduces complexity more
   than shortening a function.
2. **Every config parameter is an interface.** In config-driven research code
   (Hydra, YAML trees), each key is part of the interface of every experiment.
   Fix values that never vary and compute derived ones. A parameter that has been
   identical in every real run is not a parameter.
3. **Conventions are interfaces too.** Output directory layouts, metric key
   strings, run-naming schemes and CSV columns are informal interfaces. If more
   than one module encodes them, that's leakage. Each should have one owner.
4. **Unused generality is pure cost.** A code path no experiment exercises still
   has to be read, maintained and tested. "Somewhat general" means general across
   the uses you actually have.
5. **One-off tools rot.** Migration scripts, backfills and helpers written for a
   single event become dead weight once they've run. Archive or delete them, or
   keep them under a clearly separate name.
6. **Tests pin behaviour, and they are interfaces too.** Good tests exercise
   behaviour through deep interfaces. Tests coupled to internals make
   simplification expensive. Tests that can't fail give false confidence.
7. **Simplification is a design activity, not a cleanup chore.** Do it
   continually, design each simplification twice, and judge the result by
   whether the next change becomes easier.
