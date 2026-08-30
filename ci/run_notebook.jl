### Executes one Pluto notebook end-to-end as a plain Julia script, the same
### way a reader would get errors reported if they just ran `julia notebook.jl`.
###
### Since a Pluto notebook file's `# ╔═╡ <uuid>` cell markers are comments, the
### whole file is ordinary top-to-bottom Julia source once opened outside Pluto
### (the auto-inserted `@bind` mock macro falls back to each widget's default
### value). That means the most direct correctness check is also the simplest:
### install whatever the notebook's own `using` line asks for, then `include`
### it and let any thrown exception fail this script with a nonzero exit code
### and a full stack trace — exactly the class of bug a syntax check can't see
### (wrong function signatures, type mismatches, mismatched APIs, ...).

notebook_path = only(ARGS)

import Pkg
Pkg.activate(; temp = true)

content = read(notebook_path, String)

packages = Set{String}()
for m in eachmatch(r"(?m)^[ \t]*using[ \t]+([A-Za-z0-9_,\s]+?)[ \t]*$", content)
    for name in split(m.captures[1], ",")
        name = strip(name)
        if occursin(r"^[A-Za-z_][A-Za-z0-9_]*$", name)
            push!(packages, name)
        end
    end
end

println("Detected packages for ", notebook_path, ": ", sort(collect(packages)))
isempty(packages) || Pkg.add(collect(packages))

include(abspath(notebook_path))

println("NOTEBOOK OK: ", notebook_path)
