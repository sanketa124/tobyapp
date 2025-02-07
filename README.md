# Setting Up a New Salesforce Developer Org and Importing Metadata

This guide walks you through creating a new Salesforce Developer Org and importing metadata using Salesforce CLI.

## Prerequisites

Before you start, make sure you have the following:
- [Salesforce CLI (SFDX)](https://developer.salesforce.com/tools/sfdxcli) installed
- A [Salesforce Developer Account](https://developer.salesforce.com/signup)
- Git installed (optional, if using version control)

## Step 1: Create a New Salesforce Developer Org

1. Open a terminal and authenticate with Salesforce using SFDX:
   ```sh
   sfdx auth:web:login
   ```
   This will open a browser window for you to log in to Salesforce.

2. Once authenticated, create a new Developer Org:
   ```sh
   sfdx force:org:create -f config/project-scratch-def.json -a MyDevOrg -s
   ```
   - `-f config/project-scratch-def.json`: Path to your scratch org definition file.
   - `-a MyDevOrg`: Alias for your new org.
   - `-s`: Sets this org as the default.

3. Open the newly created org in the browser:
   ```sh
   sfdx force:org:open
   ```

## Step 2: Retrieve or Clone Metadata

```sh
git clone https://github.com/sanketa124/TobyApp.git
cd TobyApp
```

## Step 3: Push Metadata to the New Org

Once you have the metadata ready, push it to your new Developer Org:

```sh
sfdx force:source:push
```

## Step 4: Assign Permissions

If your metadata includes permission sets, assign them:

```sh
sfdx force:user:permset:assign -n YourPermissionSetName
```

## Step 5: Test and Validate

1. Run Apex tests to ensure the deployment was successful:
   ```sh
   sfdx force:apex:test:run --codecoverage
   ```

2. If necessary, check the org status:
   ```sh
   sfdx force:org:display
   ```

## Step 6: Clean Up (Optional)

If you want to delete the scratch org after testing:

```sh
sfdx force:org:delete -u MyDevOrg -p
```

## Conclusion

You have now successfully set up a new Salesforce Developer Org and imported metadata into it! 🎉

## Further Reads

For navigating to the app [App Docs](https://github.com/sanketa124/tobyapp/blob/master/docs/salesforce_app_readme.md)

For metadata explanation in the repo [Metadata Docs](https://github.com/sanketa124/tobyapp/blob/master/docs/salesforce_metadata_readme.md)

For sample data needed for testing [Sample Data](https://github.com/sanketa124/tobyapp/tree/master/data)

For more details, refer to the [Salesforce CLI Guide](https://developer.salesforce.com/docs/atlas.en-us.sfdx_cli_reference.meta/sfdx_cli_reference/).

