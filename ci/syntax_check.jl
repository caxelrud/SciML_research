### Fast gate: confirm every notebook is at least syntactically valid Julia.
### Pluto's `# ╔═╡ <uuid>` cell markers are plain comments, so a whole notebook
### file parses as ordinary top-level Julia source — this needs no package
### installs and runs in seconds, catching typos before the slow full-execution
### matrix job in run_notebook.jl bothers spinning up.

function main(notebook_paths)
    all_ok = true
    for path in notebook_paths
        src = read(path, String)
        parsed = Meta.parseall(src)
        errors = filter(e -> e isa Expr && e.head == :error, parsed.args)
        if isempty(errors)
            println("OK   ", path)
        else
            all_ok = false
            println("FAIL ", path)
            foreach(e -> println("       ", e), errors)
        end
    end
    return all_ok
end

exit(main(ARGS) ? 0 : 1)
