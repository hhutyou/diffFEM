# Run test_enzyme.jl in a separate process when Enzyme is installed in the test environment.

using Test

if Base.find_package("Enzyme") === nothing
	@info "Enzyme not available: skipping the Enzyme backend tests"
else
	@testset "Enzyme backend (separate process)" begin
		cmd = `$(Base.julia_cmd()) --startup-file=no --project=$(dirname(Base.active_project())) $(joinpath(@__DIR__, "test_enzyme.jl"))`
		@test success(pipeline(cmd; stdout = stdout, stderr = devnull))
	end
end
